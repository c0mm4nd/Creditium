import traceback
import web3
import asyncio
import logging
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from subprocess import check_output

from client import Client
from eth_keyfile import load_keyfile, decode_keyfile_json
import os
import json
from server import Prover, Lender, Treasurer


class Simulator:
    def __init__(
        self,
        deployer_addr="",
        deploy=False,
        eth=False,
        ipfs=False,
        logging_level="INFO",
    ):
        if eth:
            self.start_eth()
        if ipfs:
            self.start_ipfs()

        self.logging_level = logging_level
        self.logger = logging.Logger("sim", logging_level)

        self.deployed_contract_addrs = {
            "Treasury": None,
            "Proving": None,
            "Lending": None,
        }
        self.deployed_contracts = {
            "Treasury": None,
            "Proving": None,
            "Lending": None,
        }
        self.clients = {}
        self.lenders = {}
        self.provers = {}
        self.treasurers = {}

        self.provider = web3.HTTPProvider(endpoint_uri="http://127.0.0.1:28545")
        self.w3 = web3.Web3(self.provider, [web3.middleware.geth_poa_middleware])  # type: ignore

        self.deployer_addr = self.w3.to_checksum_address(deployer_addr)
        if deploy:
            self.deploy_contracts(deployer_addr)
            self.load_contracts(deployer_addr)
        else:
            self.load_contracts(deployer_addr)
        pass

    def start_eth(self):
        print(check_output(["docker-compose", "down"], cwd="eth").decode())
        print(check_output(["docker-compose", "up", "-d"], cwd="eth").decode())
        logging.warning("eth env started")

    def start_ipfs(self):
        print(check_output(["docker-compose", "down"], cwd="ipfs").decode())
        print(check_output(["docker-compose", "up", "-d"], cwd="ipfs").decode())
        logging.warning("ipfs env started")

    def deploy_contracts(self, deployer_addr):
        # deployer_addr = "0x71b3D7405080197fC03cA82bCDd1764F1e14ABf2"
        privkey = decode_keyfile_json(
            load_keyfile(os.path.join("eth", "keystore", deployer_addr + ".json")), ""
        )
        self.deployer_privkey = privkey

        env = os.environ.copy()
        env["PRIVATE_KEY"] = "0x" + privkey.hex()
        output = check_output(
            [
                "forge",
                "script",
                "./script/Deploy.sol",
                "--broadcast",
                "--keystore",
                f"../eth/keystore/{deployer_addr}.json",
            ],
            cwd="./contracts",
            env=env,
        )
        print("output", output.decode())

    def load_contracts(self, deployer_addr):
        privkey = decode_keyfile_json(
            load_keyfile(os.path.join("eth", "keystore", deployer_addr + ".json")), ""
        )
        self.deployer_privkey = privkey

        with open("contracts/broadcast/Deploy.sol/1231312313142/run-latest.json") as f:
            latest = json.load(f)
            for tx in latest["transactions"]:
                name = tx["contractName"]
                if name in ["Treasury", "Proving", "Lending", "KDT", "KDM"]:
                    self.deployed_contract_addrs[name] = tx["contractAddress"]

        self.deployed_contracts["Proving"] = self.w3.eth.contract(
            address=self.deployed_contract_addrs["Proving"],
            abi=json.load(open("contracts/out/Proving.sol/Proving.json"))["abi"],
        )
        self.deployed_contracts["Lending"] = self.w3.eth.contract(
            address=self.deployed_contract_addrs["Lending"],
            abi=json.load(open("contracts/out/Lending.sol/Lending.json"))["abi"],
        )
        self.deployed_contracts["Treasury"] = self.w3.eth.contract(
            address=self.deployed_contract_addrs["Treasury"],
            abi=json.load(open("contracts/out/Treasury.sol/Treasury.json"))["abi"],
        )

    def add_quorum(self, contract_type, address):
        nonce = self.w3.eth.get_transaction_count(self.deployer_addr)
        call = (
            self.deployed_contracts[contract_type]
            .functions.addQuorum(address)
            .build_transaction(
                {"chainId": 1231312313142, "from": self.deployer_addr, "nonce": nonce}
            )
        )
        signed_tx = self.w3.eth.account.sign_transaction(
            call, private_key=self.deployer_privkey
        )
        send_tx = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(send_tx)
        # print(web3.Web3.to_json(tx_receipt))
        assert tx_receipt["status"] == 1
        logging.warning(f"added {contract_type} quorum {address}")

    def set_logging_level(self, level: str):
        self.logging_level = level
        self.logger.setLevel(level)
        for obj in (
            list(self.clients.items())
            + list(self.lenders.items())
            + list(self.provers.items())
            + list(self.treasurers.items())
        ):
            obj.logger.setLevel(level)
        logging.warning(f"logging level set to {level}")

    async def repl(self):
        session = PromptSession()
        logging.info("Creditium REPL Simulator started")

        while True:
            command = ""
            with patch_stdout():
                command = await session.prompt_async(
                    ">>> ",
                    completer=WordCompleter(
                        [
                            "exit",
                            "start_eth",
                            "start_ipfs",
                            "deploy_contracts",
                            "new_client",
                            "new_prover",
                            "new_lender",
                            "new_treasurer",
                        ]
                        + ["client_" + c for c in self.clients.keys()]
                        + ["prover_" + p for p in self.provers.keys()]
                        + ["lender_" + l for l in self.lenders.keys()]
                        + ["treasurer_" + t for t in self.treasurers.keys()]
                    ),
                    auto_suggest=AutoSuggestFromHistory(),
                )
            command = command.strip()
            if command == "exit":
                logging.info("Creditium REPL Simulator exit")
                break
            await self.route(command)

    async def route(self, command):
        if command == "start_eth":
            self.start_eth()

        if command == "start_ipfs":
            self.start_ipfs()

        if command.startswith("deploy_contracts"):
            args = command.split(" ")
            if len(args) < 2:
                print("require 1 args")
                return
            deployer_addr = args[1]
            self.deploy_contracts(deployer_addr)
            self.deployer_addr = self.w3.to_checksum_address(deployer_addr)
            self.load_contracts(deployer_addr)

        if command.startswith("load_contracts"):
            args = command.split(" ")
            if len(args) < 2:
                print("require 1 args")
                return
            deployer_addr = args[1]
            self.deployer_addr = self.w3.to_checksum_address(deployer_addr)
            self.load_contracts(deployer_addr)

        if command.startswith("new_client"):
            args = command.split(" ")
            if len(args) == 3:
                address = args[1]
                client_name = args[2]
                client = Client(
                    client_name,
                    len(self.clients),
                    address,
                    proving_addr=self.deployed_contract_addrs["Proving"],
                    lending_addr=self.deployed_contract_addrs["Lending"],
                    treasury_addr=self.deployed_contract_addrs["Treasury"],
                )
                self.clients[client_name] = client
                logging.warning(f"created client {client_name}")
            if len(args) == 4:
                address = args[1]
                client_name_pre = args[2]
                count = args[3]
                for i in range(int(count)):
                    client_name = client_name_pre + str(i)
                    client = Client(
                        client_name,
                        i,
                        address,
                        proving_addr=self.deployed_contract_addrs["Proving"],
                        lending_addr=self.deployed_contract_addrs["Lending"],
                        treasury_addr=self.deployed_contract_addrs["Treasury"],
                    )
                    self.clients[client_name] = client
                    logging.warning(f"created client {client_name}")
            if len(args) not in [3, 4]:
                print("use \t`new_client <address> <prefix> <count>`")

        if command.startswith("new_prover"):
            args = command.split(" ")
            if len(args) < 3:
                print("require 2 params")
                return
            prover_address = args[1]
            prover_name = args[2]

            self.add_quorum("Proving", prover_address)
            prover = Prover(
                prover_name,
                prover_address,
                contract_addr=self.deployed_contract_addrs["Proving"],
                logging_level=self.logging_level,
            )
            self.provers[prover_name] = prover

        if command.startswith("new_lender"):
            args = command.split(" ")
            if len(args) < 3:
                print("require 2 params")
                return
            lender_address = args[1]
            lender_name = args[2]

            self.add_quorum("Lending", lender_address)
            lender = Lender(
                lender_name,
                lender_address,
                contract_addr=self.deployed_contract_addrs["Lending"],
                logging_level=self.logging_level,
            )
            self.lenders[lender_name] = lender

        if command.startswith("new_treasurer"):
            args = command.split(" ")
            if len(args) < 3:
                print("require 2 params")
                return
            treasurer_address = args[1]
            treasurer_name = args[2]

            self.add_quorum("Treasury", treasurer_address)
            treasurer = Treasurer(
                treasurer_name,
                treasurer_address,
                contract_addr=self.deployed_contract_addrs["Treasury"],
                logging_level=self.logging_level,
            )
            self.treasurers[treasurer_name] = treasurer

        if command.startswith("client_"):
            command = command.removeprefix("client_")
            args = command.split(" ")
            name = args[0]
            fn = args[1]
            client = self.clients[name]
            await getattr(client, fn)(*args[2:])  # dont need self param

        if command.startswith("all_client"):
            command = command.removeprefix("all_client").strip()
            args = command.split(" ")
            fn = args[0]
            try:
                await asyncio.gather(
                    *[
                        getattr(client, fn)(*args[1:])
                        for client in self.clients.values()
                    ]
                )  # dont need self param
            except Exception as e:
                print(e)
                traceback.print_exc()

        if command.startswith("clear_client"):
            self.clients.clear()

        if command.startswith("prover_"):
            command = command.removeprefix("prover_")
            args = command.split(" ")
            name = args[0]
            fn = args[1]
            prover = self.provers[name]
            await getattr(prover, fn)(*args[2:])  # dont need self param

        if command.startswith("lender_"):
            command = command.removeprefix("lender_")
            args = command.split(" ")
            name = args[0]
            fn = args[1]
            lender = self.lenders[name]
            await getattr(lender, fn)(*args[2:])  # dont need self param

        if command.startswith("clear_lender"):
            self.lenders.clear()

        if command.startswith("treasurer_"):
            command = command.removeprefix("treasurer_")
            args = command.split(" ")
            name = args[0]
            fn = args[1]
            treasurer = self.treasurers[name]
            await getattr(treasurer, fn)(*args[2:])  # dont need self param

        if command.startswith("logging_"):
            command = command.removeprefix("logging_")
            level = command.upper()
            self.set_logging_level(level)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Creditium Simulator",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-a", "--deployer_addr", default="0x71b3D7405080197fC03cA82bCDd1764F1e14ABf2"
    )
    parser.add_argument("-d", "--deploy", action="store_true")
    parser.add_argument("--eth", action="store_true")
    parser.add_argument("--ipfs", action="store_true")
    args = parser.parse_args()

    sim = Simulator(
        deployer_addr=args.deployer_addr,
        deploy=args.deploy,
        eth=args.eth,
        ipfs=args.ipfs,
    )
    loop = asyncio.new_event_loop()
    loop.run_until_complete(sim.repl())
