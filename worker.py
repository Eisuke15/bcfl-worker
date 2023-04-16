import torch
from hexbytes import HexBytes
from torch.utils.data import DataLoader
from web3 import Web3
from web3.types import EventData

from net import Net

from training import train, test

class Worker:
    def __init__(self, index, contract_abi, contract_address, trainset, testset) -> None:
        self.index = index

        # training assets
        self.train_loader = DataLoader(trainset, batch_size = 256, shuffle = True, num_workers = 2, pin_memory=True)
        self.test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)

        # contract
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        self.account = self.w3.eth.accounts[self.index]
        self.w3.eth.default_account = self.account
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)

        # for simulation
        self.submitted_model_count = 0

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
            model = torch.load(model_path, map_location=self.device)
            for key in aggregated_model:
                aggregated_model[key] += model[key] / len(CIDs)

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
        
        train(model=self.net, optimizer=self.optimizer, device=self.device, train_loader=self.train_loader, num_epochs=10, progress_bar=False)
        test(model=self.net, device=self.device, test_loader=self.test_loader, progress_bar=False)


    def upload_model(self):
        """upload model, returns CID."""
        cid = f"{self.index}_{self.submitted_model_count}" # for simulation. dummy value.
        torch.save(self.net.state_dict(), f"models/{cid}.pth")
        self.submitted_model_count += 1
        return cid
    

    def workers_to_vote(self, latest_model_index: int) -> list:
        if latest_model_index == 0:
            return []
        else:
            return [self.contract.functions.models(latest_model_index - 1).call()[0]] # for simulaion. dummy value.
    

    def submit(self, CID: str, latest_model_index: int) -> HexBytes:
        """Submit model CID to the contract. Returns tx_hash."""
        tx_hash = self.contract.functions.submitModel(CID, self.workers_to_vote(latest_model_index)).transact()
        print(f"worker {self.index} submitted model")
        return tx_hash
    
    def handle_event(self, event: EventData) -> HexBytes:
        """Handle event."""
        latest_model_index = event['args']['latestModelIndex']
        cids_to_aggregate = self.get_votable_models_CIDs(latest_model_index)
        self.aggregate(cids_to_aggregate)
        self.train()
        cid = self.upload_model()
        tx_hash = self.submit(cid, latest_model_index)
        return tx_hash