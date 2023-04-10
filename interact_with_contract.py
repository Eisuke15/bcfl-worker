import json

from web3 import Web3


abi_file = "FederatedLearning.json"
contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load the contract's ABI
with open(abi_file) as f:
    contract_abi = json.load(f)['abi']


class Worker:
    def __init__(self, index, w3) -> None:
        self.index = index
        self.w3 = w3


def init_worker(i):
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

    account = w3.eth.accounts[i]
    w3.eth.default_account = account
    return Worker(i, w3)


def register(worker: Worker):
    # Get the contract object
    contract = worker.w3.eth.contract(abi=contract_abi, address=contract_address)

    # https://web3py.readthedocs.io/en/v5/contracts.html#web3.contract.ContractFunction.transact
    tx_hash = contract.functions.register().transact()
    tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
    processed_logs = contract.events.LearningRightGranted().process_receipt(tx_receipt)
    print(processed_logs)

    print(f"worker {worker.index} registered")
    

if __name__ == "__main__":
    workers = [init_worker(i) for i in range(10)]
    for w in workers:
        register(w)
