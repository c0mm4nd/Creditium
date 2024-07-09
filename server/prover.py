import json
import traceback
import web3
import asyncio
import aioipfs
import hybrid_pke
import binascii
import logging


class Prover:
    def __init__(self, name, prover_addr, contract_addr, logging_level="INFO") -> None:
        self.name = name
        self.logging_level = logging_level
        self.logger = logging.Logger(name)
        self.logger.setLevel(logging_level)

        self.provider = web3.HTTPProvider(endpoint_uri="http://127.0.0.1:28545")
        self.w3 = web3.Web3(self.provider, [web3.middleware.geth_poa_middleware])
        self.prover_addr = prover_addr
        with open(f"./eth/keystore/{prover_addr}.json") as keyfile:
            encrypted_key = keyfile.read()
            self.private_key = self.w3.eth.account.decrypt(encrypted_key, "")

        self.contract_addr = contract_addr
        self.Proving = self.w3.eth.contract(
            address=contract_addr,
            abi=json.load(open("contracts/out/Proving.sol/Proving.json"))["abi"],
        )

        self.ipfs = aioipfs.AsyncIPFS(maddr="/ip4/127.0.0.1/tcp/15001")
        self.hpke = hybrid_pke.default()
        self.info = b""  # shared metadata, correspondance-level
        self.aad = b""  # shared metadata, message-level
        self.task = {}

    async def host_secret(self, secret, pubkey):
        encap, encrypted = self.hpke.seal(pubkey, self.info, self.aad, secret)
        content = json.dumps(
            {
                "encap": binascii.hexlify(encap).decode(),
                "encrypted": binascii.hexlify(encrypted).decode(),
            }
        ).encode()
        entry = await self.ipfs.core.add_bytes(content)
        cid = entry["Name"]  # or entry["Hash"]
        return cid

    async def response(self, rid, cid):
        nonce = self.w3.eth.get_transaction_count(self.prover_addr)
        call = self.Proving.functions.response(
            rid, cid.encode()  # convert cid to bytes
        ).build_transaction(
            {"chainId": 1231312313142, "from": self.prover_addr, "nonce": nonce}
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.private_key
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        assert tx_receipt["status"] == 1
        self.logger.warning(f"{self.name}: response {rid} with cid {cid}")

    async def brainless_approve(self, rid, respID):
        nonce = self.w3.eth.get_transaction_count(self.prover_addr)
        call = self.Proving.functions.approve(rid, respID).build_transaction(
            {"chainId": 1231312313142, "from": self.prover_addr, "nonce": nonce}
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
        event_filter_EventRequest = self.Proving.events.EventRequest.create_filter(
            fromBlock="latest"
        )
        while True:
            for req in event_filter_EventRequest.get_new_entries():
                self.logger.info(
                    f"{self.name}: listen a new request", req
                )  # {'args': {'id': 2, 'user': '0x4B54772Cc9e233B4fe1F04A65a994aF40A6834ac', 'reqContent': b'=M;\xbd\xea\xab\xcd\x13{\xfeCO\xb8u\x9eI\xe9\x8e\xe0\xcc\x87\xd6\xb868B\x02\xe07\xa2\xcfG'}, 'event': 'EventRequest', 'logIndex': 0, 'transactionIndex': 0, 'transactionHash': HexBytes('0x73a90ccf316265735013772b112cbce02cd18a562fbeebaebce44ac79a67e790'), 'address': '0x245a3ba7c2F8428A70D4A32a3606324724bA4a12', 'blockHash': HexBytes('0x5cd03cb3dfce42082cd8903fb8d6874226a093b8cfa22e8b208d8088cdc61709'), 'blockNumber': 3124}
                req_content = json.loads(req["args"]["reqContent"])
                pubkey = binascii.unhexlify(req_content["pubkey"])  # ensure bytes
                secret = "TODO: fake personal content here".encode()  # TODO
                cid = await self.host_secret(secret, pubkey)
                self.logger.info(f"{self.name}: uploaded to ipfs")
                await self.response(req["args"]["id"], cid)

            await asyncio.sleep(1)

    async def listen_EventResponse(self):
        event_filter_EventResponse = self.Proving.events.EventResponse.create_filter(
            fromBlock="latest"
        )
        while True:
            for resp in event_filter_EventResponse.get_new_entries():
                # do an approve here
                self.logger.info(f"{self.name}: listen a new response", resp)
                await self.brainless_approve(resp["args"]["id"], resp["args"]["respID"])
            await asyncio.sleep(1)

    async def listen_EventDone(self):
        event_filter_EventDone = self.Proving.events.EventResponse.create_filter(
            fromBlock="latest"
        )
        while True:
            for req in event_filter_EventDone.get_new_entries():
                self.logger.info(f"{self.name}: listen a new done")
            await asyncio.sleep(1)
