from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from common.net import CNN_v4 as Net
from common.training import test, testset, train, trainset
import random

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('-n', '--n-round', type=int, help="number of rounds to train for", default=100000)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

filter = 'iid'
indices=torch.load(f'./indices/{filter}.pt')


subsets = [Subset(trainset, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True) for subset in subsets]
test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

models = [Net().to(device) for _ in range(n_node)]
optimizers = [Adam(model.parameters()) for model in models]

submited_1 = Net().to(device).state_dict()
submited_2 = Net().to(device).state_dict()
submited_3 = Net().to(device).state_dict()


with open(f'logs/afl_{filter}.log', 'w') as f:
    for round in range(args.n_round):
        print(f"Round {round+1}")

        randomly_selected_worker = random.randint(0, n_node - 1)

        model = models[randomly_selected_worker]
        optimizer = optimizers[randomly_selected_worker]
        train_loader = train_loaders[randomly_selected_worker]

        print(f"Worker {randomly_selected_worker}")

        global_model_dict = Net().to(device).state_dict()
        
        for key in global_model_dict:
            global_model_dict[key] = (submited_1[key] + submited_2[key] + submited_3[key]) / 3

        global_model = Net().to(device)
        global_model.load_state_dict(global_model_dict)

        acc = test(model=global_model, device=device, test_loader=test_loader)

        print(acc, file=f)
        f.flush()

        model.load_state_dict(global_model_dict)

        # train models
        train(model=model, optimizer=optimizer, device=device, train_loader=train_loader, num_epochs=5)

        submited_3 = submited_2.copy()
        submited_2 = submited_1.copy()
        submited_1 = model.state_dict()
