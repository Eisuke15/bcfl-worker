import torch
from hexbytes import HexBytes
from torch.utils.data import DataLoader
from web3 import Web3
from web3.types import EventData

from common.net import CNN as Net

from common.training import train, test
from torch.nn import init
import torch.nn as nn

torch.backends.cudnn.benchmark = True

class Worker:
    def __init__(self, index, contract_abi, contract_address, trainset, gpu_num = 0, progress_bar=False, num_epochs=5, malicious=False, val_lazy=False, agg_lazy=False) -> None:
        self.index = index

        # training assets
        self.train_loader = DataLoader(trainset, batch_size = 256, shuffle = True, num_workers = 2, pin_memory=True)
        self.device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.progress_bar = progress_bar
        self.num_epochs = num_epochs
        self.malicious = malicious
        self.val_lazy = val_lazy
        self.agg_lazy = agg_lazy

        # contract
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545", request_kwargs={"timeout": 100}))
        self.account = self.w3.eth.accounts[self.index]
        self.w3.eth.default_account = self.account
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)

        # constants
        self.votable_model_num = self.contract.functions.VotableModelNum().call()
        self.initial_model_cid = self.contract.functions.initialModelCID().call()

        # for simulation
        self.submitted_model_count = 0
        self.total_gas_used = 0


    def register(self) -> HexBytes:
        """Register the worker to the contract."""
        
        tx_hash = self.contract.functions.register().transact({
            'gas': 1000000,
        })
        return tx_hash

    def download_net(self, CID: str) -> str:
        """与えられたCIDのモデルをダウンロードし、そのパスを返す。"""
        return f"models/{CID}.pth"

    def aggregate(self, CIDs: list) -> str:
        """与えられたCIDのモデルと、自身のモデルを平均化し、ロードする"""
        current_net = self.net.state_dict()
        aggregated_model = current_net.copy()

        for CID in CIDs:
            model_path = self.download_net(CID)
            model = torch.load(model_path, map_location=self.device)
            for key in aggregated_model:
                aggregated_model[key] = aggregated_model[key] + (model[key] - current_net[key]) / (len(CIDs) + 1)

        self.net.load_state_dict(aggregated_model)

    
    def get_recent_model_CIDs(self, latest_model_index: int, num: int) -> list:
        """与えられたmodel indexから遡ってnum個のモデルのCIDを取得する。"""

        indices = range(max(latest_model_index - num, 0), latest_model_index)
        return [self.contract.functions.models(i).call()[0] for i in indices]

    def train(self):
        """学習を行う。"""  
        train(model=self.net, optimizer=self.optimizer, device=self.device, train_loader=self.train_loader, num_epochs=self.num_epochs, progress_bar=self.progress_bar)


    def upload_model(self):
        """upload model, returns CID."""
        cid = f"{self.index}_{self.submitted_model_count}" # for simulation. dummy value.
        torch.save(self.net.state_dict(), f"models/{cid}.pth")
        self.submitted_model_count += 1
        return cid
    
    def score_recent_models(self, latest_model_index: int, num: int) -> list:
        cids = self.get_recent_model_CIDs(latest_model_index, num)
        paths = [self.download_net(cid) for cid in cids]
        models = [Net().to(self.device) for path in paths]
        for model, path in zip(models, paths):
            model.load_state_dict(torch.load(path, map_location=self.device))
        scores = [test(model, self.device, self.train_loader, progress_bar=False) for model in models]
        return cids, scores
    

    def submit(self, CID: str, cids_to_vote: list[str]) -> HexBytes:
        """Submit model CID to the contract. Returns tx_hash."""
        tx_hash = self.contract.functions.submitModel(CID, cids_to_vote).transact()
        return tx_hash
    
    def handle_event(self, event: EventData) -> HexBytes:
        """Handle event."""
        latest_model_index = event['args']['latestModelIndex']

        cids, scores = self.score_recent_models(latest_model_index, self.votable_model_num)

        # 最もスコアの良いモデルのcidを取得
        if scores:
            best_score_index = scores.index(max(scores))
            cids_to_vote = [cids[best_score_index]]
        else:
            cids_to_vote = []

        base_model_cid = cids_to_vote[0] if cids_to_vote else self.initial_model_cid
        self.net.load_state_dict(torch.load(self.download_net(base_model_cid), map_location=self.device))

        if not self.malicious:
            self.train()

            if not self.agg_lazy:
                if self.val_lazy:
                    cids_to_aggregate = cids[:2]
                else:
                    # スコアが0.8を超えるモデルの先頭2つを集計対象とする
                    cids_to_aggregate = [cid for cid, score in zip(cids, scores) if score > 0.5][:2]
                self.aggregate(cids_to_aggregate)
        else:
            def init_weights(m):
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.normal_(m.weight, mean=0, std=1000)
                    if m.bias is not None:
                        init.zeros_(m.bias)
            # でたらめな値でモデルを初期化する
            self.net.apply(init_weights)
        
        cid = self.upload_model()
        tx_hash = self.submit(cid, cids_to_vote)
        gasUsed = self.get_gas_used(tx_hash)
        self.total_gas_used += gasUsed
        return tx_hash, cid
    
    def get_token_balance(self):
        """get token balance"""
        return self.contract.functions.balanceOf(self.account).call()
    
    def get_gas_used(self, tx_hash: HexBytes):
        """get gas used"""
        return self.w3.eth.wait_for_transaction_receipt(tx_hash).gasUsed
