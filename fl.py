from argparse import ArgumentParser

import torch
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from net import Net
from training import test, train

parser = ArgumentParser()
parser.add_argument('-n', '--n-round', type=int, help="number of rounds to train for", default=100)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

filter = 'iid'
indices=torch.load(f'./indices/{filter}.pt')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])
dataset_train = MNIST(root='data', train=True, download=True, transform=transform)
subsets = [Subset(dataset_train, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2) for subset in subsets]
testset = MNIST(root = 'data', train = False, download = True, transform = transform)
test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

models = [Net().to(device) for _ in range(n_node)]
optimizers = [Adam(model.parameters(), lr=0.0001) for model in models]

accuracy = [[] for _ in range(n_node + 1)] 

global_model = Net().to(device).state_dict()

for round in range(args.n_round):
    print(f"Round {round+1}")

    for i, (model, optimizer, train_loader) in enumerate(zip(models, optimizers, train_loaders)):
        print(f"Worker {i}")
        model.load_state_dict(global_model)

        # train models
        train(model=model, optimizer=optimizer, device=device, train_loader=train_loader, num_epochs=10)

        # test models
        acc = test(model=model, device=device, test_loader=test_loader)
        accuracy[i].append(acc)
        print(f"Worker {i} accuracy: {acc}")

    new_global_model = Net().to(device).state_dict()

    # aggregate models
    for model in models:
        model_parameters = model.state_dict()
        for key in new_global_model:
            new_global_model[key] += model_parameters[key]
    for key in new_global_model:
        new_global_model[key] /= n_node

    global_model = new_global_model

    # test global model
    global_model_test = Net().to(device)
    global_model_test.load_state_dict(global_model)
    acc = test(model=global_model_test, device=device, test_loader=test_loader)
    accuracy[-1].append(acc)
    print(f"Global model accuracy: {acc}")


torch.save(accuracy, f'graph/fl_{filter}.pt')
        