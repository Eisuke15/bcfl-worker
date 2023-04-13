import sys
import time

import torch
import torchvision
from torch.utils.data.dataset import Subset
from web3.exceptions import ContractLogicError
from web3.logs import IGNORE

from config import CONTRACT_ABI, CONTRACT_ADDRESS
from worker import Worker


def register_and_watch(i: int):
    """`i`th worker register to the contract and watch the event."""

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

    indices = torch.load('indices/iid.pt')
    trainset = Subset(trainset, indices[0])
    worker = Worker(i, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, testset)

    try:
        worker.register()
        print(f"worker {i} successfully registered")
    except ContractLogicError:
        print(f"worker {i} already registered")


    event_filter = worker.contract.events.LearningRightGranted.create_filter(fromBlock="latest", argument_filters={'client': worker.w3.eth.default_account})

    while True:
        for event in event_filter.get_new_entries():
            try:
                tx_hash = worker.handle_event(event)
                # tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
                # logs = worker.contract.events.LearningRightGranted().process_receipt(tx_receipt, errors=IGNORE)
                print(f"worker {i} successfully handled the event")
                
            except ContractLogicError as e:
                print(f"worker {i} missed the chance.")
                # print(e)

        time.sleep(5)


if __name__ == "__main__":
    args = sys.argv
    register_and_watch(int(args[1]))
