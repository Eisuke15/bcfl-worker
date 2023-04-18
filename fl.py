from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from net import CNN_v4 as Net
from training import test, train

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('-n', '--n-round', type=int, help="number of rounds to train for", default=100)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

filter = 'r00_s01'
indices=torch.load(f'./indices_cifar10/{filter}.pt')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(0.5, 0.5,)])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5), 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)
testset = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

subsets = [Subset(trainset, indices[i]) for i in range(n_node)]
train_loaders = [DataLoader(subset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True) for subset in subsets]
test_loader = DataLoader(testset, batch_size = 256, shuffle = False, num_workers = 2, pin_memory=True)

models = [Net().to(device) for _ in range(n_node)]
optimizers = [Adam(model.parameters()) for model in models]

accuracy = [[] for _ in range(n_node + 1)] 

global_model = Net().to(device).state_dict()

for round in range(args.n_round):
    print(f"Round {round+1}")

    for i, (model, optimizer, train_loader) in enumerate(zip(models, optimizers, train_loaders)):
        print(f"Worker {i}")
        model.load_state_dict(global_model)

        # train models
        train(model=model, optimizer=optimizer, device=device, train_loader=train_loader, num_epochs=5)

        # test models
        acc = test(model=model, device=device, test_loader=test_loader)
        accuracy[i].append(acc)
        print(f"Worker {i} accuracy: {acc}")

    new_global_model = global_model.copy()

    # aggregate models
    for model in models:
        model_parameters = model.state_dict()
        for key in new_global_model:
            new_global_model[key] = new_global_model[key] + (model_parameters[key] - global_model[key]) / n_node
   
    global_model = new_global_model

    # test global model
    global_model_test = Net().to(device)
    global_model_test.load_state_dict(global_model)
    acc = test(model=global_model_test, device=device, test_loader=test_loader)
    accuracy[-1].append(acc)
    print(f"Global model accuracy: {acc}")


torch.save(accuracy, f'graph_cifar10/fl_{filter}.pt')
        