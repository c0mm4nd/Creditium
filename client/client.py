from eth_typing import Address
import web3
import asyncio
import aioipfs
import numpy as np
from eth_keyfile import load_keyfile, decode_keyfile_json
import os
import hybrid_pke
import tensorflow as tf
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import json
import binascii
import logging

from utils import (
    generate_EMNIST_writer_based_data,
    generate_bal_private_data,
    load_EMNIST_data,
    generate_alignment_data,
    generate_partial_data,
    remove_last_layer,
    dict_to_bytes,
    bytes_to_dict,
    load_MNIST_data,
)


class Client:
    def __init__(
        self,
        name,
        idx, # REMOVEME
        self_addr,
        proving_addr,
        lending_addr,
        treasury_addr,
        batch_size=5,
        epochs=5,
        logging_level="INFO"
    ):
        self.idx = idx # REMOVEME
        self.name = name
        self.logging_level = logging_level
        self.logger = logging.Logger(name)
        self.logger.setLevel(logging_level)
        
        self.addr = self_addr
        self.privkey = decode_keyfile_json(
            load_keyfile(os.path.join("eth", "keystore", self.addr + ".json")), ""
        )

        # init providers
        self.provider = web3.HTTPProvider(endpoint_uri="http://127.0.0.1:18545")
        self.w3 = web3.Web3(self.provider, [web3.middleware.geth_poa_middleware])  # type: ignore
        self.ipfs = aioipfs.AsyncIPFS(maddr="/ip4/127.0.0.1/tcp/15001")

        # ensure the checksum address
        self.addr = self.w3.to_checksum_address(self.addr)

        self.proving_addr = self.w3.to_checksum_address(proving_addr)
        self.lending_addr = self.w3.to_checksum_address(lending_addr)
        self.treasury_addr = self.w3.to_checksum_address(treasury_addr)
        self.Proving = self.w3.eth.contract(
            address=self.proving_addr,
            abi=json.load(open("contracts/out/Proving.sol/Proving.json"))["abi"],
        )
        self.Lending = self.w3.eth.contract(
            address=self.lending_addr,
            abi=json.load(open("contracts/out/Lending.sol/Lending.json"))["abi"],
        )
        self.Treasury = self.w3.eth.contract(
            address=self.treasury_addr,
            abi=json.load(open("contracts/out/Treasury.sol/Treasury.json"))["abi"],
        )

        self.hpke = hybrid_pke.default()  # type: ignore
        self.info = b""  # shared metadata, correspondance-level
        self.aad = b""  # shared metadata, message-level

        # FL
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size

        N_parties = 10
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

        (
            X_train_EMNIST,
            y_train_EMNIST,
            X_test_EMNIST,
            y_test_EMNIST,
            writer_ids_train,
            writer_ids_test,
        ) = load_EMNIST_data(
            "./dataset/emnist-letters.mat",
            standarized=True,
        )

        y_train_EMNIST += len(public_classes)
        y_test_EMNIST += len(public_classes)

        all_private_data, _ = generate_bal_private_data(
            X_train_EMNIST,
            y_train_EMNIST,
            N_parties=N_parties,
            classes_in_use=private_classes,
            N_samples_per_class=N_samples_per_class,
            data_overlap=False,
        )
        self.private_data = all_private_data[idx]

        X_tmp, y_tmp = generate_partial_data(
            X=X_test_EMNIST,
            y=y_test_EMNIST,
            class_in_use=private_classes,
        )
        self.private_test_data = {"X": X_tmp, "y": y_tmp}

        X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST = load_MNIST_data(
            standarized=True
        )

        self.public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}

        self.model_tensors = {
            "logits": None,
            "classifier": None,
            "weights": None,  # model.get_weights(),
        }

        self.handler_pubkeys = {}

    async def close(self):
        await self.ipfs.close()

    async def proving_request_and_wait(self):
        onetime_privkey, onetime_pubkey = self.hpke.generate_key_pair()

        nonce = self.w3.eth.get_transaction_count(self.addr)
        req_content = json.dumps(
            {"pubkey": binascii.hexlify(onetime_pubkey).decode()}
        ).encode()
        call_function = self.Proving.functions.request(req_content).build_transaction(
            {"chainId": 1231312313142, "from": self.addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.privkey
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1  # 0 = failed

        # extract rid from the log
        logs = self.Proving.events.EventRequest().process_receipt(tx_receipt)
        rid = logs[0]["args"]["id"]
        self.logger.info(f"{self.name} proving rid: {rid}")

        # listen EventDone
        event_filter_EventDone = self.Proving.events.EventDone.create_filter(  # type: ignore
            fromBlock="latest"
        )
        while True:
            self.logger.debug(f"waiting for done")
            for done in event_filter_EventDone.get_new_entries():
                if done["args"]["id"] == rid:
                    result = done["args"]["respContent"]
                    secret = await self.read_secret(result, onetime_privkey)
                    self.logger.warning(f"{self.name} received secret from the prover")
                    self.logger.debug(secret.decode())
                    return secret
            await asyncio.sleep(1)

    async def lending_request_and_wait(self):
        nonce = self.w3.eth.get_transaction_count(self.addr)
        req_content = json.dumps(
            {
                "provingTx": "0x",  # TODO: link the proving txhash in prod
            }
        ).encode()
        call_function = self.Lending.functions.request(req_content).build_transaction(
            {"chainId": 1231312313142, "from": self.addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.privkey
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1  # 0 = failed

        # extract rid from the log
        logs = self.Lending.events.EventRequest().process_receipt(tx_receipt)
        rid = logs[0]["args"]["id"]
        self.logger.info(f"{self.name} lending rid: {rid}")

        # listen HandleRequest
        handler_pubkey = None
        event_filter_HandleRequest = self.Lending.events.HandleRequest.create_filter(fromBlock="latest")  # type: ignore
        exit_event_filter_HandleRequest = False
        while True:
            self.logger.debug(f"{self.name} waiting for HandleRequest")
            for hand in event_filter_HandleRequest.get_new_entries():
                self.logger.info(f"{self.name} received new HandleRequest event")
                if hand["args"]["rid"] == rid:
                    handler = hand["args"]["handler"]
                    self.logger.info(
                        f"{self.name} handler {handler} start handling my request"
                    )
                    handler_pubkey = hand["args"]["pubkey"]
                    self.handler_pubkeys[rid] = handler_pubkey
                    handler_model_cid = hand["args"]["initModel"]

                    await self.init_model(
                        rid,
                        handler,
                        self.handler_pubkeys[rid],
                        handler_model_cid,
                    )

                    exit_event_filter_HandleRequest = True
            if exit_event_filter_HandleRequest:
                break
            await asyncio.sleep(1)

        # listen HandlerUpdateModel
        event_filter_HandlerUpdateModel = self.Lending.events.HandlerUpdateModel.create_filter(fromBlock="latest")  # type: ignore
        event_filter_EventDone = self.Lending.events.EventDone.create_filter(fromBlock="latest")  # type: ignore
        while True:
            self.logger.debug(f"{self.name} waiting for HandlerUpdateModel and EventDone")
            for handler_update in event_filter_HandlerUpdateModel.get_new_entries():
                if handler_update["args"]["rid"] == rid:
                    handler = handler_update["args"]["handler"]
                    self.logger.info(f"{self.name} got handler {handler} update notif")
                    round = handler_update["args"]["round"]
                    self.logger.info(f"{self.name} round {round}")
                    handler_model_cid = handler_update["args"]["model"]
                    handler_pubkey = self.handler_pubkeys[rid]

                    asyncio.create_task(
                        self.update_model(
                            rid,
                            handler,
                            handler_pubkey,
                            handler_model_cid,
                        )
                    )
            for done in event_filter_EventDone.get_new_entries():
                if done["args"]["id"] == rid:
                    # result = done["args"]["respContent"].decode()  # from bytes to str
                    # secret = await self.read_secret(result, onetime_privkey)
                    self.logger.warning(f"{self.name} {rid} done")
                    return
            await asyncio.sleep(1)

    async def try_get_token_from_treasury(self, amount: str):
        amount = int(amount)
        nonce = self.w3.eth.get_transaction_count(self.addr)
        amount_in_bytes = amount.to_bytes(32, 'big')
        call_function = self.Treasury.functions.request(amount_in_bytes).build_transaction(
            {"chainId": 1231312313142, "from": self.addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.privkey
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1  # 0 = failed

        # extract rid from the log
        logs = self.Treasury.events.EventRequest().process_receipt(tx_receipt)
        rid = logs[0]["args"]["id"]
        self.logger.info(f"{self.name} treasury rid: {rid}")

        # listen EventDone
        event_filter_Borrow = self.Treasury.events.Borrow.create_filter(  # type: ignore
            fromBlock="latest"
        )
        event_filter_Reject = self.Treasury.events.Reject.create_filter(  # type: ignore
            fromBlock="latest"
        )

        while True:
            self.logger.debug(f"waiting for borrow or reject")
            for borrow in event_filter_Borrow.get_new_entries():
                if borrow["args"]["rid"] == rid:
                    self.logger.warning(f"{self.name} received {amount} token from the treasury")
                    return
            for reject in event_filter_Reject.get_new_entries():
                if reject["args"]["rid"] == rid:
                    self.logger.warning(f"{self.name} is rejected by treasury")
                    return
            await asyncio.sleep(1)

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

    async def read_secret(self, cid: bytes, privkey: bytes) -> bytes:
        json_content: bytes = await self.ipfs.core.cat(cid.decode())  # type: ignore
        content = json.loads(json_content.decode())
        encap = binascii.unhexlify(content["encap"])
        encrypted = binascii.unhexlify(content["encrypted"])
        secret_bytes = self.hpke.open(encap, privkey, self.info, self.aad, encrypted)
        return secret_bytes
    
    async def read_public(self, cid: bytes) -> bytes:
        public_bytes: bytes = await self.ipfs.core.cat(cid.decode())  # type: ignore
        return public_bytes

    async def init_model(
        self,
        rid: int,
        handler: Address,
        handler_pubkey: bytes,
        handler_model_cid: bytes,
    ):
        self.logger.warning(f"{self.name} start init client model")
        model_weights_bytes = await self.read_public(handler_model_cid)
        # real updating
        handler_model = bytes_to_dict(model_weights_bytes)

        model_A_twin = clone_model(handler_model["classifier"])
        model_A_twin.set_weights(handler_model["classifier"].get_weights())
        model_A_twin.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.logger.debug(f"{self.name} start full stack training ... ")

        model_A_twin.fit(
            self.private_data["X"],
            self.private_data["y"],
            batch_size=32,
            epochs=25,
            shuffle=True,
            verbose=0,
            validation_data=[self.private_test_data["X"], self.private_test_data["y"]],
            callbacks=[
                EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10)
            ],
        )

        self.logger.debug(f"{self.name} full stack training done")

        model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

        self.model_tensors = {
            "logits": model_A,
            "classifier": model_A_twin,
            "weights": model_A_twin.get_weights(),
        }

        user_updated_model = dict_to_bytes(self.model_tensors)
        updated_model_cid = await self.host_secret(user_updated_model, handler_pubkey)

        # call borrowerUpdate
        nonce = self.w3.eth.get_transaction_count(self.addr)
        call_function = self.Lending.functions.borrowerUpdate(
            rid, [handler], [updated_model_cid]  # TODO: support multi-handler
        ).build_transaction(
            {"chainId": 1231312313142, "from": self.addr, "nonce": nonce}
        )

        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.privkey
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.info(f"{self.name} done init_model")

    async def update_model(
        self,
        rid: int,
        handler: Address,
        handler_pubkey: bytes,
        handler_model_cid: bytes,
    ):
        model_weights_bytes = await self.read_public(handler_model_cid)
        # real updating
        handler_model = bytes_to_dict(model_weights_bytes)

        alignment_data = generate_alignment_data(
            self.public_dataset["X"], self.public_dataset["y"], self.N_alignment
        )
        self.logger.info(f"{self.name} starting alignment with public logits... ")

        weights_to_use = self.model_tensors["weights"]

        self.model_tensors["logits"].set_weights(weights_to_use)
        self.model_tensors["logits"].fit(
            alignment_data["X"],
            handler_model["logits"],
            batch_size=256,  # self.logits_matching_batchsize,
            epochs=1,  # self.N_logits_matching_round,
            shuffle=True,
            verbose=0,
        )
        self.model_tensors["weights"] = self.model_tensors["logits"].get_weights()
        self.logger.info(f"{self.name} model done alignment")

        self.logger.info("model starting training with private data... ")
        weights_to_use = self.model_tensors["weights"]
        self.model_tensors["classifier"].set_weights(weights_to_use)
        self.model_tensors["classifier"].fit(
            self.private_data["X"],
            self.private_data["y"],
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=0,
        )

        self.model_tensors["weights"] = self.model_tensors["classifier"].get_weights()
        self.logger.info(f"{self.name} model done private training.")

        user_updated_model = dict_to_bytes(self.model_tensors)
        updated_model_cid = await self.host_secret(user_updated_model, handler_pubkey)

        # call borrowerUpdate
        nonce = self.w3.eth.get_transaction_count(self.addr)
        call_function = self.Lending.functions.borrowerUpdate(
            rid, [handler], [updated_model_cid]  # TODO: support multi-handler
        ).build_transaction(
            {"chainId": 1231312313142, "from": self.addr, "nonce": nonce}
        )
        # Sign transaction
        signed_tx = self.w3.eth.account.sign_transaction(
            call_function, private_key=self.privkey
        )
        # Send transaction
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        # Wait for transaction receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.info(f"{self.name} done update_model")
