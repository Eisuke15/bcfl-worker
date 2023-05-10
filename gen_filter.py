import os
import random
from argparse import ArgumentParser

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

parser = ArgumentParser()
parser.add_argument('--bs', type=int, help="calculation batch size", default=32)
parser.add_argument('--seed', type=int, help='random seed', default=1)
parser.add_argument('--ratio', type=int, help='noniid ratio (%)', default=90)
parser.add_argument('--nnodes', type=int, help='number of nodes', default=10)
parser.add_argument('--iid', help='iid dataset', default=False, action='store_true')
parser.add_argument('--inbalanced', help='inbalanced dataset', default=False, action='store_true')
args = parser.parse_args()

random.seed(args.seed)

dir = './indices'

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=2)

indices = [[] for i in range (args.nnodes)]

index = 0
num_labels = [[0] * 10 for _ in range(args.nnodes)]
for data in trainloader :
    _, y = data
    y = y.tolist()

    for i, label in enumerate(y):
        global_index = index + i

        if args.iid:
            n = random.randint(0, args.nnodes - 1)
            indices[n].append(global_index)
            num_labels[n][label] += 1

        elif args.inbalanced:
            n = random.choice([i for i in range(10) for _ in range(i+1)])
            indices[n].append(global_index)
            num_labels[n][label] += 1
            
        else:
            # 指定した確率で素直にlabel番目のノードに割り振る。
            if random.randint(0, 99) < args.ratio:
                indices[label].append(global_index)
                num_labels[label][label] += 1

            # そうでない場合はランダムにlabel番目以外のノードに割り振る。
            else:
                if args.inbalanced:
                    n = random.choice()
                # 0 ~ label-1, label+1 ~ nnodeまでの整数の内からランダムに一つ選ぶ
                n = random.choice([j for j in range(0, args.nnodes) if not j == label])
                indices[n].append(global_index)
                num_labels[n][label] += 1

    index += len(y)


print([len(i) for i in indices])

filename=os.path.join(dir, 'iid.pt' if args.iid else 'inbalanced_iid.pt' if args.inbalanced else f'r{args.ratio:02d}_s{args.seed:02d}.pt')
torch.save(indices,filename)
print('Done')

print(num_labels)
labels = [
    'Worker',
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
    'Summary'
]

print(*labels, sep=',')
for i, l in enumerate(num_labels):
    print(*([i] + l + [sum(l)]), sep=',')

sum_class = [sum([l[i] for l in num_labels]) for i in range(10)]
print(*(['Summary'] + sum_class+ [sum(sum_class)]), sep=',')


