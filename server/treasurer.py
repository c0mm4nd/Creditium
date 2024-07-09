import asyncio
import binascii
import traceback
import web3
import json
import aioipfs
import hybrid_pke
import logging

from utils.nn import bytes_to_dict, cnn_2layer_fc_model


class Treasurer:
    def __init__(
        self, name, treasurer_addr, contract_addr, logging_level="INFO"
    ) -> None:
        self.name = name
        self.logging_level = logging_level
        self.logger = logging.Logger(name)
        self.logger.setLevel(logging_level)

        self.provider = web3.HTTPProvider(endpoint_uri="http://127.0.0.1:28545")
        self.w3 = web3.Web3(self.provider, [web3.middleware.geth_poa_middleware])
        self.treasurer_addr = treasurer_addr
        with open(f"./eth/keystore/{treasurer_addr}.json") as keyfile:
            encrypted_key = keyfile.read()
            self.private_key = self.w3.eth.account.decrypt(encrypted_key, "")

        self.contract_addr = contract_addr
        self.Treasury = self.w3.eth.contract(
            address=contract_addr,
            abi=json.load(open("contracts/out/Treasury.sol/Treasury.json"))["abi"],
        )

        self.ipfs = aioipfs.AsyncIPFS(maddr="/ip4/127.0.0.1/tcp/15001")
        self.hpke = hybrid_pke.default()
        self.info = b""  # shared metadata, correspondance-level
        self.aad = b""  # shared metadata, message-level
        self.task = {}

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

        self.model = cnn_2layer_fc_model(**self.model_params)

    async def load_public_model(self):
        model_cid_bytes = self.Treasury.functions.get_model().call()
        model_bytes = self.read_public(model_cid_bytes)
        self.model = bytes_to_dict(model_bytes)

    async def read_public(self, cid: bytes) -> bytes:
        public_bytes: bytes = await self.ipfs.core.cat(cid.decode())  # type: ignore
        return public_bytes

    async def response(self, rid, cid):
        nonce = self.w3.eth.get_transaction_count(self.treasurer_addr)
        call = self.Treasury.functions.response(rid, cid).build_transaction(
            {"chainId": 1231312313142, "from": self.treasurer_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.warning(f"{self.name}: response {rid}")

    async def brainless_approve(self, rid, respID):
        nonce = self.w3.eth.get_transaction_count(self.treasurer_addr)
        call = self.Treasury.functions.approve(rid, respID).build_transaction(
            {"chainId": 1231312313142, "from": self.treasurer_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.warning(
            f"{self.name}: (brainless_)approve call status {tx_receipt['status']}"
        )

    async def listen(self):
        self.t = asyncio.create_task(self.listen_web3())
        self.logger.warning(f"{self.name} is now listening")

    async def listen_web3(self):
        # e.g. {"args": {"id": 2, "enctyptedRealWorldIdentification": "you guess my id"}, "event": "NewRequest", "logIndex": 0, "transactionIndex": 0, "transactionHash": "0xb50d7cd3e09e7f81a6bbb527b80b884e9e6d4a17f5530594e49e7d2842f53a8d", "address": "0x08abf1bC52427fe8b648cB6d70a7c646ef0C4411", "blockHash": "0x251dea5e27fd82c67818d959e7215fcb1c7ebfe707fd6e83720a93bde16bf90e", "blockNumber": 1440}
        try:
            await asyncio.gather(
                self.listen_EventRequest(),
                self.listen_EventResponse(),
                self.listen_EventDone(),
            )
        except Exception as e:
            print(e)
            traceback.print_exc()

    async def listen_EventRequest(self):
        event_filter_EventRequest = self.Treasury.events.EventRequest.create_filter(
            fromBlock="latest"
        )
        while True:
            for req in event_filter_EventRequest.get_new_entries():
                self.logger.info(f"{self.name}: listen a new request", req)

                # run eval and return the result in bytes
                # y_pred = self.model["classifier"].predict(self.private_test_data["X"]).argmax(axis = 1)

                ok = True
                respContent = 0x1.to_bytes(32, "big") if ok else 0x0.to_bytes(32, "big")

                self.logger.info(f"{self.name}: uploaded to ipfs")
                await self.response(req["args"]["id"], respContent)

            await asyncio.sleep(1)

    async def listen_EventResponse(self):
        event_filter_EventResponse = self.Treasury.events.EventResponse.create_filter(
            fromBlock="latest"
        )
        while True:
            for resp in event_filter_EventResponse.get_new_entries():
                # do an approve here
                self.logger.info(f"{self.name}: listen a new response", resp)
                await self.brainless_approve(resp["args"]["id"], resp["args"]["respID"])
            await asyncio.sleep(1)

    async def listen_EventDone(self):
        event_filter_EventDone = self.Treasury.events.EventResponse.create_filter(
            fromBlock="latest"
        )
        while True:
            for req in event_filter_EventDone.get_new_entries():
                self.logger.info(f"{self.name}: listen a new done")
            await asyncio.sleep(1)
