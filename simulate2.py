import random
import sys
import time

import torch
from torch.utils.data.dataset import Subset
from web3.exceptions import ContractLogicError
from torch.utils.data import DataLoader

from config import CONTRACT_ABI, CONTRACT_ADDRESS
from training import test, testset, trainset
from worker import Worker
import random


def simulate():
    """Simulate workers. Next worker to submit model is chosen randomly."""



    indices = torch.load('indices/iid.pt')
    workers = [Worker(i, CONTRACT_ABI, CONTRACT_ADDRESS, Subset(trainset, indices[i]), gpu_num=1, progress_bar=True) for i in range(10)]
    for w in workers:
        print(len(w.train_loader.dataset))
    test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

    for w in workers:
        w.register()
        print(f"worker {w.index} successfully registered")

    with open(f"logs/all.log", "w") as f:
        while True:
            worker_id = random.randint(0, 9)
            worker = workers[worker_id]


            _, _, learning_right, latest_model_index = worker.contract.functions.clientInfo(worker.account).call()
            if not learning_right:
                continue
            else:
                print(f"worker {worker_id} got the right. latest model index: {latest_model_index}")
                mock_event = {'args': {'latestModelIndex': latest_model_index}}
                worker.handle_event(mock_event)
                acc = test(model=worker.net, test_loader=test_loader, device=worker.device, progress_bar=True)
                balance = worker.get_token_balance()
                total_gas_used = worker.total_gas_used
                print(f"worker {worker_id} submitted model accuracy: {acc} balance: {balance}: totalGasUsed: {total_gas_used}")
                print(worker_id, latest_model_index, acc, balance, total_gas_used, file=f)
                f.flush()


if __name__ == "__main__":
    args = sys.argv
    simulate()
