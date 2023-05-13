import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('filter', type=str, help='what dataset to use')
args = parser.parse_args()
filter = args.filter

num_transaction = 1000

# plot acc_list
fig, ax = plt.subplots()

ax.set_ylim(0.4, 0.9)
ax.yaxis.grid(True)
ax.set_xlabel('Transaction')
ax.set_ylabel('Accuracy')

x = [[] for _ in range(11)]
y = [[] for _ in range(11)]
with open(f'logs/{filter}.log') as f:
    lines = f.readlines()
    for line in lines:
        worker_id, index, acc, balance, total_gas_used = line.split()
        worker_id = int(worker_id)
        index = int(index)
        acc = float(acc)
        balance = float(balance)
        total_gas_used = float(total_gas_used)
        if index <= num_transaction:
            x[worker_id].append(index)
            y[worker_id].append(acc)

for i in range(10):
    ax.plot(x[i], y[i], label=f'Worker {i}')


ax.plot(x[-1], y[-1], label='Aggregated model', color='black')
print(max(y[-1]))
  
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(f'graph/bcfl_{filter}.pdf')

