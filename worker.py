import torch
import torch.nn as nn
import torchvision
from hexbytes import HexBytes
from torch.utils.data import DataLoader
from tqdm import tqdm
from web3 import Web3
from web3.types import EventData

from net import Net


class Worker:
    def __init__(self, index, contract_abi, contract_address) -> None:
        self.index = index

        # training assets
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transform)
        self.train_loader = DataLoader(trainset, batch_size = 256, shuffle = True, num_workers = 2, pin_memory=True)
        testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)
        self.test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

        # contract
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        account = self.w3.eth.accounts[self.index]
        self.w3.eth.default_account = account
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)

    def register(self) -> HexBytes:
        """Register the worker to the contract."""
        
        tx_hash = self.contract.functions.register().transact()
        print(f"worker {self.index} registered")

        return tx_hash

    def download_net(self, CID: str) -> str:
        """与えられたCIDのモデルをダウンロードし、そのパスを返す。"""
        return f"models/{CID}.pth"

    def aggregate(self, CIDs: list) -> str:
        """与えられたCIDのモデルを平均化し、ロードする。"""
        aggregated_model = Net().to(self.device).state_dict()
        

        for CID in CIDs:
            model_path = self.download_net(CID)
            model = torch.load(model_path)
            for key in aggregated_model:
                aggregated_model[key] = aggregated_model[key] + (model[key] - aggregated_model[key]) / len(CIDs)

        self.net.load_state_dict(aggregated_model)

    
    def get_votable_models_CIDs(self, latest_model_index: int) -> list:
        """与えられたmodel indexから遡ってVotableModelNum個のモデル（VotableModelNumに満たない場合はFLの初期モデルも加える。）のCIDを取得する。"""
        
        votable_model_num = self.contract.functions.VotableModelNum().call()
        indices = range(max(latest_model_index - votable_model_num, 0), latest_model_index)
        cids = [self.contract.functions.models(i).call()[0] for i in indices]
        
        if len(cids) < votable_model_num:
            cids = [self.contract.functions.initialModelCID().call()] + cids

        return cids


    def train(self):
        """学習を行う。"""  
        criterion = nn.CrossEntropyLoss()

        for epoch in range(3):
            # training
            sum_correct = 0

            for (inputs, labels) in tqdm(self.train_loader, desc=f"epoch:{epoch+1} training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = outputs.max(1)
                sum_correct += (predicted == labels).sum().item()

            accuracy = float(sum_correct/len(self.train_loader.dataset))
            print(f"epoch:{epoch+1} train accuracy={accuracy}")

            sum_correct = 0

            # validation
            for (inputs, labels) in tqdm(self.test_loader, desc=f"epoch:{epoch+1} testing", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                _, predicted = outputs.max(1)
                sum_correct += (predicted == labels).sum().item()
            
            accuracy = float(sum_correct/len(self.test_loader.dataset))
            print(f"epoch:{epoch+1} test accuracy={accuracy}")


    def upload_model(self, latest_model_index: int):
        """upload model, returns CID."""
        torch.save(self.net.state_dict(), f"models/{latest_model_index}.pth")
        return latest_model_index
    

    def workers_to_vote(self, latest_model_index: int) -> list:
        return [] if latest_model_index == 0 else [str(latest_model_index - 1)]
    

    def submit(self, CID: str, latest_model_index: int) -> HexBytes:
        """Submit model CID to the contract. Returns tx_hash."""
        tx_hash = self.contract.functions.submitModel(str(latest_model_index), self.workers_to_vote(latest_model_index)).transact()
        print(f"worker {self.index} submitted model")
        return tx_hash
    
    def handle_event(self, event: EventData) -> HexBytes:
        """Handle event."""
        latest_model_index = event['args']['latestModelIndex']
        cids = self.get_votable_models_CIDs(latest_model_index)
        self.aggregate(cids)
        self.train()
        cid = self.upload_model(latest_model_index)
        tx_hash = self.submit(cid, latest_model_index)
        return tx_hash