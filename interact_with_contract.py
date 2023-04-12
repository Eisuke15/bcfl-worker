import json

from worker import Worker


abi_file = "FederatedLearning.json"
contract_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load the contract's ABI
with open(abi_file) as f:
    contract_abi = json.load(f)['abi']


def watch_event(worker: Worker):
    print(f"Watch event {worker.index}")
    contract = worker.w3.eth.contract(abi=contract_abi, address=contract_address)
    print(worker.w3.eth.default_account)
    event_filter = contract.events.LearningRightGranted.create_filter(fromBlock="latest", argument_filters={'client': worker.w3.eth.default_account})
    for event in event_filter.get_new_entries():
        print(event)


def simulate():
    workers = [Worker(i, contract_abi, contract_address) for i in range(10)]
    for w in workers:
        tx_hash = w.register()
        tx_receipt = w.w3.eth.get_transaction_receipt(tx_hash)
        logs = w.contract.events.LearningRightGranted().process_receipt(tx_receipt)
    
    for i in range(10):
        event = logs[-1]
        for w in workers:
            if w.w3.eth.default_account == event['args']['client']:
                worker = w
                print(f"next worker is {w.index}")
                break
        tx_hash = worker.handle_event(event)
        tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
        logs = worker.contract.events.LearningRightGranted().process_receipt(tx_receipt)


if __name__ == "__main__":
    simulate()
