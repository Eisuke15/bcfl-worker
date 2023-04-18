from config import CONTRACT_ABI, CONTRACT_ADDRESS
from training import test, testset, trainset
from worker import Worker

worker = Worker(0, CONTRACT_ABI, CONTRACT_ADDRESS, trainset, testset, gpu_num=1)
cids = [worker.contract.functions.initialModelCID().call()]
for i in range(10000):
    try:
        cids.append(worker.contract.functions.models(i).call()[0])
    except Exception: 
        break

for latest_model_index in range(5, len(cids)):
    worker.aggregate(cids[latest_model_index - 5:latest_model_index])
    acc = test(model=worker.net, test_loader=worker.test_loader, device=worker.device, progress_bar=True)
    print(f'{latest_model_index}: {acc}')
