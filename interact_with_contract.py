import json

from web3 import Web3
from web3.logs import STRICT, IGNORE, DISCARD, WARN


abi_file = "FederatedLearning.json"
contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load the contract's ABI
with open(abi_file) as f:
    contract_abi = json.load(f)['abi']


class Worker:
    def __init__(self, index, w3: Web3) -> None:
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
    print(f"worker {worker.index} registered")

    tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
    processed_logs = contract.events.LearningRightGranted().process_receipt(tx_receipt)

    return processed_logs



def watch_event(worker: Worker):
    print(f"Watch event {worker.index}")
    contract = worker.w3.eth.contract(abi=contract_abi, address=contract_address)
    print(worker.w3.eth.default_account)
    event_filter = contract.events.LearningRightGranted.create_filter(fromBlock="latest", argument_filters={'client': worker.w3.eth.default_account})
    for event in event_filter.get_new_entries():
        print(event)


def submit_model(worker, model_index):
    contract = worker.w3.eth.contract(abi=contract_abi, address=contract_address)
    import datetime
    tx_hash = contract.functions.submitModel(str(model_index), [] if model_index == 0 else [str(model_index - 1)]).transact()
    print(f"worker {worker.index} submitted model")
    tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
    processed_logs = contract.events.LearningRightGranted().process_receipt(tx_receipt, errors=IGNORE)

    return processed_logs


def simulate():
    workers = [init_worker(i) for i in range(10)]
    for w in workers:
        logs = register(w)

    for i in range(5):
        event_args = logs[-1]['args']
        for w in workers:
            if w.w3.eth.default_account == event_args['client']:
                next_worker = w
                break
        model_index = event_args['modelIndex']
        logs = submit_model(next_worker, model_index)


if __name__ == "__main__":
    simulate()
