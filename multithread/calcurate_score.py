import torch
from torch.utils.data import DataLoader

from config import CONTRACT_ABI, CONTRACT_ADDRESS
from common.training import test, testset, trainset
from worker import Worker

worker = Worker(0, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, progress_bar=True)
cids = [worker.contract.functions.initialModelCID().call()]

test_loader = DataLoader(testset, batch_size = 128, shuffle = True, num_workers = 2, pin_memory=True)

acc_list = []
for i in range(2000):
    try:
        cid, author_index = worker.contract.functions.models(i).call()
        cids.append(cid)
        print(cid)
    except Exception: 
        break

for latest_model_index in range(len(cids)):
    worker.aggregate(cids[max(0, latest_model_index - 10):latest_model_index])
    acc = test(model=worker.net, test_loader=test_loader, device=worker.device, progress_bar=True)
    acc_list.append(acc)
    print(f'{latest_model_index}: {acc}')

torch.save(acc_list, 'graph/acc_list.pth')