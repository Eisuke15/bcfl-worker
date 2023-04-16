from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from net import Net
from training import test, train

parser = ArgumentParser()
parser.add_argument('-n', '--n-round', type=int, help="number of rounds to train for", default=10)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

indices=torch.load('./indices/iid.pt')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
dataset_train = MNIST(root='data', train=True, download=True, transform=transform)
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2) for subset in subsets]
testset = MNIST(root = 'data', train = False, download = True, transform = transform)
test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

models = [Net().to(device) for _ in range(n_node)]
optimizers = [Adam(model.parameters(), lr=0.0001) for model in models]
criterion = nn.CrossEntropyLoss()

accuracy = [[] for _ in range(n_node + 1)] 

for round in range(args.n_round + 1):
    print(f"Round {round+1}")
    global_model = Net().to(device).state_dict()

    # aggregate models
    for model in models:
        model_parameters = model.state_dict()
        for key in global_model:
            global_model[key] += model_parameters[key] / n_node
    
    print("Global model accracy")
    acc = test(model=model, device=device, test_loader=test_loader)
    accuracy[-1].append(acc)

    # train models
    for i, (model, optimizer, train_loader) in enumerate(zip(models, optimizers, train_loaders)):
        print(f"Worker {i}")
        acc = test(model=model, device=device, test_loader=test_loader)
        accuracy[i].append(acc)
        model.load_state_dict(global_model)
        train(model=model, optimizer=optimizer, device=device, criterion=criterion, train_loader=train_loader)


torch.save(accuracy, 'graph/fl.pt')
        