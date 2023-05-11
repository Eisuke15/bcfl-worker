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

    filter = 'iid'

    indices = torch.load(f'indices/{filter}.pt')
    workers = [Worker(i, CONTRACT_ABI, CONTRACT_ADDRESS, Subset(trainset, indices[i]), gpu_num=1, progress_bar=True) for i in range(10)]
    for w in workers:
        print(len(w.train_loader.dataset))

    worker_for_aggregate = Worker(10, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, gpu_num=1, progress_bar=True)

    test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)
    acc_list = []
    cids = []

    for w in workers:
        w.register()
        print(f"worker {w.index} successfully registered")

    with open(f"logs/{filter}.log", "w") as f:
        while True:
            worker_id = random.randint(0, 9)
            worker = workers[worker_id]

            _, _, learning_right, latest_model_index = worker.contract.functions.clientInfo(worker.account).call()
            if not learning_right:
                continue
            else:
                print(f"worker {worker_id} got the right. latest model index: {latest_model_index}")
                mock_event = {'args': {'latestModelIndex': latest_model_index}}
                _, cid = worker.handle_event(mock_event)
                acc = test(model=worker.net, test_loader=test_loader, device=worker.device, progress_bar=True)
                balance = worker.get_token_balance()
                total_gas_used = worker.total_gas_used
                print(f"worker {worker_id} submitted model accuracy: {acc} balance: {balance}: totalGasUsed: {total_gas_used}")
                print(worker_id, latest_model_index, acc, balance, total_gas_used, file=f)
                f.flush()

                cids.append(cid)
                latest_cid_index = len(cids)
                worker_for_aggregate.aggregate(cids[max(0, latest_cid_index - 10):latest_cid_index])
                acc = test(model=worker_for_aggregate.net, test_loader=test_loader, device=worker_for_aggregate.device, progress_bar=True)
                acc_list.append(acc)
                print(f"global acc: {acc}")
                print(-1, latest_cid_index, acc, -1, -1, file=f)


if __name__ == "__main__":
    args = sys.argv
    simulate()
