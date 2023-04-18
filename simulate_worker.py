import random
import sys
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Subset
from web3.exceptions import ContractLogicError
from web3.logs import IGNORE

from config import CONTRACT_ABI, CONTRACT_ADDRESS
from training import test, trainset, testset
from worker import Worker


def register_and_watch(i: int):
    """`i`th worker register to the contract and watch the event."""

    random.seed(i)

    indices = torch.load('indices/r00_s01.pt')
    train_subset = Subset(trainset, indices[i])
    worker = Worker(i, CONTRACT_ABI, CONTRACT_ADDRESS, train_subset, testset)

    try:
        worker.register()
        print(f"worker {i} successfully registered")
    except (ContractLogicError, ValueError):
        print(f"worker {i} already registered")


    event_filter = worker.contract.events.LearningRightGranted.create_filter(fromBlock="latest", argument_filters={'client': worker.w3.eth.default_account})
    with open(f"logs/worker_{i}.log", "w") as f:
        while True:
            new_entries = list(event_filter.get_new_entries())
            if new_entries:
                event = new_entries[-1]
                try:
                    latest_model_index = event['args']['latestModelIndex']
                    print(f"worker {i} got the right. latest model index: {latest_model_index}")
                    worker.handle_event(event)
                    acc = test(model=worker.net, test_loader=worker.test_loader, device=worker.device, progress_bar=False)
                    print(f"worker {i} submitted model accuracy: {acc}")
                    print(latest_model_index, acc, file=f)
                    f.flush()
                    
                except (ContractLogicError, ValueError):
                    print(f"worker {i} missed the chance.")

            time.sleep(random.uniform(0, 2))


if __name__ == "__main__":
    args = sys.argv
    register_and_watch(int(args[1]))
