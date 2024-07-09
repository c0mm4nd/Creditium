import json
from typing import Dict
import web3
import aioipfs
import asyncio
import hybrid_pke
import binascii
import traceback
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model
from tensorflow.keras.saving import serialize_keras_object, deserialize_keras_object
import logging
from utils import (
    generate_alignment_data,
    cnn_2layer_fc_model,
    load_MNIST_data,
    # train_models,
    remove_last_layer,
    bytes_to_dict,
    dict_to_bytes,
    train_model,
)


class Lender:
    def __init__(self, name, lender_addr, contract_addr, logging_level="INFO") -> None:
        self.name = name
        self.logging_level = logging_level
        self.logger = logging.Logger(name)
        self.logger.setLevel(logging_level)

        self.provider = web3.HTTPProvider(endpoint_uri="http://127.0.0.1:28545")
        self.w3 = web3.Web3(self.provider, [web3.middleware.geth_poa_middleware])  # type: ignore

        self.contract_addr = self.w3.to_checksum_address(contract_addr)
        self.Lending = self.w3.eth.contract(
            address=contract_addr,
            abi=json.load(open("contracts/out/Lending.sol/Lending.json"))["abi"],
        )

        self.lender_addr = self.w3.to_checksum_address(lender_addr)
        with open(f"./eth/keystore/{self.lender_addr}.json") as keyfile:
            encrypted_key = keyfile.read()
            self.private_key = self.w3.eth.account.decrypt(encrypted_key, "")

        self.ipfs = aioipfs.AsyncIPFS(maddr="/ip4/127.0.0.1/tcp/5001")

        self.hpke = hybrid_pke.default()  # type: ignore
        self.info = b""  # shared metadata, correspondance-level
        self.aad = b""  # shared metadata, message-level

        private_classes = [10, 11, 12, 13, 14, 15]
        public_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        N_samples_per_class = 3
        N_alignment = 3000

        n_classes = len(public_classes) + len(private_classes)

        self.N_alignment = N_alignment
        self.model_params = {
            "n_classes": n_classes,
            "input_shape": (28, 28),
            "n1": 128,
            "n2": 256,
            "dropout_rate": 0.2,
        }

        empty_model = cnn_2layer_fc_model(**self.model_params)
        # TODO: load current_model from oracle network

        X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST = load_MNIST_data(
            standarized=True
        )

        self.public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}

        # pre-train the self.current_model
        train_model(
            empty_model,
            X_train_MNIST,
            y_train_MNIST,
            X_test_MNIST,
            y_test_MNIST,
            early_stopping=True,
            **{
                "min_delta": 0.001,
                "patience": 3,
                "batch_size": 128,
                "epochs": 20,
                "is_shuffle": True,
                "verbose": 1,
            },
        )

        self.current_model = {
            "logits": remove_last_layer(empty_model, loss="mean_absolute_error"),
            "classifier": empty_model,
            "weights": empty_model.get_weights(),  # model.get_weights(),
        }

        # REMOVEME: testing the serializing
        _test = dict_to_bytes(self.current_model)
        _test = bytes_to_dict(_test)

        # clear me per train
        self.train_rid_onetime_keypairs = {}  # rid ->key_pair

        # clear me per round
        self.round_rid_models = {}

    async def host_secret(self, secret: bytes, pubkey: bytes) -> bytes:
        encap, encrypted = self.hpke.seal(pubkey, self.info, self.aad, secret)
        content = json.dumps(
            {
                "encap": binascii.hexlify(encap).decode(),
                "encrypted": binascii.hexlify(encrypted).decode(),
            }
        ).encode()
        entry: dict[str, str] = await self.ipfs.core.add_bytes(content)  # type: ignore
        cid = entry["Name"].encode()  # or entry["Hash"]
        return cid

    async def host_public(self, public: bytes) -> bytes:
        entry: dict[str, str] = await self.ipfs.core.add_bytes(public)  # type: ignore
        cid = entry["Name"].encode()  # or entry["Hash"]
        return cid

    async def read_secret(self, cid: bytes, privkey: bytes) -> bytes:
        json_content: bytes = await self.ipfs.core.cat(cid.decode())  # type: ignore
        content = json.loads(json_content.decode())
        encap = binascii.unhexlify(content["encap"])
        encrypted = binascii.unhexlify(content["encrypted"])
        secret_bytes = self.hpke.open(encap, privkey, self.info, self.aad, encrypted)
        return secret_bytes

    async def aggregate(self):
        self.logger.warning(f"{self.name} start aggreagting")
        alignment_data = generate_alignment_data(
            self.public_dataset["X"], self.public_dataset["y"], self.N_alignment
        )

        logits = 0
        # update logits
        self.logger.debug("update logits ... ")
        for user_model in self.round_rid_models.values():
            user_model["logits"].set_weights(user_model["weights"])
            logits += user_model["logits"].predict(alignment_data["X"], verbose=0)
        logits /= len(self.round_rid_models)

        self.current_model = {
            "logits": logits,  # the avergated logit
            "classifier": self.current_model["classifier"],
            "weights": self.current_model["weights"],
        }
        safetensor_bytes = dict_to_bytes(self.current_model)

        rid_models = {}
        for rid in self.train_rid_onetime_keypairs.keys():
            model = await self.host_public(safetensor_bytes)
            rid_models[rid] = model

        # call handlerUpdate
        nonce = self.w3.eth.get_transaction_count(self.lender_addr)
        call = self.Lending.functions.handlerUpdate(
            list(rid_models.keys()), list(rid_models.values())
        ).build_transaction(
            {"chainId": 1231312313142, "from": self.lender_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.warning(f"{self.name} aggregated, clearing")

        # clean round info
        self.logger.warning(f"{self.name} clear round tensors")
        self.round_rid_models.clear()

        # processing receipt to get event
        handler_update_logs = self.Lending.events.HandlerUpdateModel().process_receipt(
            tx_receipt
        )
        response_logs = self.Lending.events.EventResponse().process_receipt(tx_receipt)
        if len(handler_update_logs) > 0:
            return  # do not clear
        if len(response_logs) > 0:
            # the EventResponse is triggered, clear all user data
            self.train_rid_onetime_keypairs.clear()
        self.logger.warning(f"{self.name} all cache clear")

    async def handle(self, rid, onetime_pubkey):
        init_model_bytes = dict_to_bytes(self.current_model)
        init_model_cid = await self.host_public(init_model_bytes)

        nonce = self.w3.eth.get_transaction_count(self.lender_addr)
        call_function = self.Lending.functions.handle(
            rid,
            # self.lender_addr,
            onetime_pubkey,
            init_model_cid,
        ).build_transaction(
            {"chainId": 1231312313142, "from": self.lender_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.warning(f"{self.name}: handled {rid}")

    async def brainless_approve(self, rid, respID):
        nonce = self.w3.eth.get_transaction_count(self.lender_addr)
        call = self.Lending.functions.approve(rid, respID).build_transaction(
            {"chainId": 1231312313142, "from": self.lender_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        self.logger.warning(
            f"{self.name}: (brainless_)approve call status {tx_receipt['status']}"
        )

    async def close(self):
        await self.ipfs.close()

    async def listen(self):
        self.t = asyncio.create_task(self.listen_web3())
        self.logger.warning(f"{self.name} is now listening")

    async def listen_web3(self):
        # loop.run_until_complete( listen_web3())
        try:
            await asyncio.gather(
                self.listen_EventRequest(),
                self.listen_HandleRequest(),
                self.listen_BorrowerUpdateModel(),
                self.listen_HandlerUpdateModel(),
                self.listen_EventResponse(),
                self.listen_EventDone(),
            )
        except Exception as e:
            print(e)
            traceback.print_exc()

    async def listen_EventRequest(self):
        # https://web3py.readthedocs.io/en/stable/filters.html
        event_filter_EventRequest = self.Lending.events.EventRequest.create_filter(  # type: ignore
            fromBlock="latest"
        )

        while True:
            for req in event_filter_EventRequest.get_new_entries():
                # handle the request
                if len(self.train_rid_onetime_keypairs) >= 10:
                    continue
                rid: int = req["args"]["id"]

                req_content_bytes: bytes = req["args"]["reqContent"]
                req_content = json.loads(req_content_bytes.decode())

                proving_tx = req_content["provingTx"]  # TODO: verify me

                onetime_privkey, onetime_pubkey = self.hpke.generate_key_pair()
                self.train_rid_onetime_keypairs[rid] = (onetime_privkey, onetime_pubkey)

                asyncio.create_task(self.handle(rid, onetime_pubkey))

            await asyncio.sleep(1)

    async def listen_HandleRequest(self):
        # https://web3py.readthedocs.io/en/stable/filters.html
        event_filter_HandleRequest = self.Lending.events.HandleRequest.create_filter(  # type: ignore
            fromBlock="latest"
        )

        while True:
            for h in event_filter_HandleRequest.get_new_entries():
                # ignore this event
                pass
            await asyncio.sleep(1)

    async def listen_BorrowerUpdateModel(self):
        # https://web3py.readthedocs.io/en/stable/filters.html
        event_filter_BorrowerUpdateModel = (
            self.Lending.events.BorrowerUpdateModel.create_filter(fromBlock="latest")
        )

        while True:
            for event in event_filter_BorrowerUpdateModel.get_new_entries():
                # aggragate model when self in event.handlers
                if self.lender_addr == event["args"]["handler"]:
                    rid = event["args"]["rid"]
                    assert rid in self.train_rid_onetime_keypairs.keys()

                    model_cid = event["args"]["model"]
                    (priv, pub) = self.train_rid_onetime_keypairs[rid]
                    model_bytes = await self.read_secret(model_cid, priv)
                    model = bytes_to_dict(model_bytes)
                    self.round_rid_models[rid] = model
                    # if all(10) client's weights ready, start aggregating
                    if len(self.round_rid_models) == 10:
                        asyncio.create_task(self.aggregate())  # not blocking
                    else:
                        self.logger.warning(
                            f"not ready to agg, count is {len(self.round_rid_models.keys())}"
                        )
            await asyncio.sleep(1)

    async def listen_HandlerUpdateModel(self):
        # https://web3py.readthedocs.io/en/stable/filters.html
        event_filter_HandlerUpdateModel = (
            self.Lending.events.HandlerUpdateModel.create_filter(fromBlock="latest")
        )  # type: ignore

        while True:
            for event in event_filter_HandlerUpdateModel.get_new_entries():
                # ignore this
                pass
            await asyncio.sleep(1)

    async def listen_EventResponse(self):
        event_filter_EventResponse = self.Lending.events.EventResponse.create_filter(  # type: ignore
            fromBlock="latest"
        )
        while True:
            for resp in event_filter_EventResponse.get_new_entries():
                self.logger.warning(f"{self.name}: listen a new response", resp)
                await self.brainless_approve(resp["args"]["id"], resp["args"]["respID"])
            await asyncio.sleep(1)

    async def listen_PublishModel(self):
        event_filter_PublishModel = self.Lending.events.PublishModel.create_filter(  # type: ignore
            fromBlock="latest"
        )
        while True:
            for p in event_filter_PublishModel.get_new_entries():
                # update local models
                rid = p["args"]["rid"]
                model_cid = p["args"]["model"]
                (priv, pub) = self.train_rid_onetime_keypairs[rid]
                secret = await self.read_secret(model_cid, priv)
                # TODO: use safetensor convert it into a model
                pass
            await asyncio.sleep(1)

    async def listen_EventDone(self):
        event_filter_EventDone = self.Lending.events.EventDone.create_filter(  # type: ignore
            fromBlock="latest"
        )
        while True:
            for req in event_filter_EventDone.get_new_entries():
                # TODO: remove local model when others already published
                pass
            await asyncio.sleep(1)
