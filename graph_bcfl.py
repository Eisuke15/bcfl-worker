import torch
import matplotlib.pyplot as plt

acc_list = torch.load('graph/bcfl_noniid_aggregated_acc.pth')

# plot acc_list
fig, ax = plt.subplots()
ax.plot(acc_list[:1000], label='Aggregated model', color='black')
ax.set_ylim(0.4, 0.9)
ax.yaxis.grid(True)
ax.set_xlabel('Transaction')
ax.set_ylabel('Accuracy')


for i in range(10):
    with open('logs/worker_' + str(i) + '.log') as f:
        lines = f.readlines()
        x = []
        y = []
        for line in lines:
            index, acc = line.split()
            index = int(index)
            acc = float(acc)
            if index <= 1000:
                x.append(index)
                y.append(acc)
    ax.plot(x, y, label=f'Worker {i}')

  
plt.legend()
plt.savefig(f'graph/acc_list.png')
plt.savefig(f'graph/acc_list.pdf')

