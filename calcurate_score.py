from config import CONTRACT_ABI, CONTRACT_ADDRESS
from training import test, testset, trainset
from worker import Worker
import torch

worker = Worker(0, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, testset, gpu_num=1)
cids = [worker.contract.functions.initialModelCID().call()]

acc_list = []
for i in range(10000):
    try:
        cid, author_index = worker.contract.functions.models(i).call()
        cids.append(cid)
    except Exception: 
        break

for latest_model_index in range(len(cids)):
    worker.aggregate(cids[max(0, latest_model_index - 10):latest_model_index])
    acc = test(model=worker.net, test_loader=worker.test_loader, device=worker.device, progress_bar=True)
    acc_list.append(acc)
    print(f'{latest_model_index}: {acc}')

torch.save(acc_list, 'graph/acc_list.pth')