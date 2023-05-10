from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST

from net import CNN_v4 as Net
from training import test, testset, train, trainset

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()
parser.add_argument('-n', '--n-round', type=int, help="number of rounds to train for", default=100)
parser.add_argument('-g', '--gpu-num', type=int, help='what gpu to use', default=0)
args = parser.parse_args()

n_node = 10

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")

filter = 'r00_s01'
indices=torch.load(f'./indices/{filter}.pt')


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
        train(model=model, optimizer=optimizer, device=device, train_loader=train_loader, num_epochs=1)

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


torch.save(accuracy, f'graph/fl_{filter}.pt')
        