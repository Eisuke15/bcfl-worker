from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

parser = ArgumentParser()
parser.add_argument('filter', type=str, help='what dataset to use')
args = parser.parse_args()
filter = args.filter

lim = 200

accuracy = torch.load(f'graph/fl_{filter}.pt')
fig, ax = plt.subplots()

for i, acc in enumerate(accuracy):
    ax.plot(acc[:lim],  label=f'Worker {i}')

ax.plot(accuracy[-1][:lim], label='Global model', color='black')

ax.set_ylim(0.4, 0.9)

# horizontal grid line
ax.yaxis.grid(True)

ax.set_xlabel('Round')
ax.set_ylabel('Accuracy')

plt.legend(fontsize="small")

# reduce margin
plt.tight_layout()
plt.savefig(f'graph/fl_{filter}.pdf')
