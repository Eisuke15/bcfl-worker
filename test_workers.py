import torch
import torchvision
from torch.utils.data import Subset
from web3.logs import IGNORE

from worker import Worker
from config import CONTRACT_ABI, CONTRACT_ADDRESS


def simulate():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

    indices = torch.load('indices/iid.pt')

    trainsets = [Subset(trainset, indices[i]) for i in range(10)]

    workers = [Worker(i, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, testset) for i, trainset in enumerate(trainsets)]

    for w in workers:
        tx_hash = w.register()
        tx_receipt = w.w3.eth.get_transaction_receipt(tx_hash)
        logs = w.contract.events.LearningRightGranted().process_receipt(tx_receipt, errors=IGNORE)
    
    for i in range(1000):
        print('------------------')
        print(i)
        event = logs[-1]
        for w in workers:
            if w.w3.eth.default_account == event['args']['client']:
                worker = w
                print(f"next worker is {w.index}")
                break
        tx_hash = worker.handle_event(event)
        tx_receipt = worker.w3.eth.get_transaction_receipt(tx_hash)
        logs = worker.contract.events.LearningRightGranted().process_receipt(tx_receipt, errors=IGNORE)


if __name__ == "__main__":
    simulate()