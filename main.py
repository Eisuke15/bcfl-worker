import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

BATCH_SIZE = 256
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 10
PATH = './data'
NUM_WORKERS = 2 if torch.cuda.is_available() else 0

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root = PATH, train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory=True)

testset = torchvision.datasets.MNIST(root = PATH, train = False, download = True, transform = transform)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_WORKERS, pin_memory=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,32,3)

        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist 

for epoch in range(EPOCH):
    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計

    for (inputs, labels) in tqdm(trainloader, desc=f"epoch:{epoch+1} training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()

    mean_loss = sum_loss*BATCH_SIZE/len(trainloader.dataset)
    accuracy = float(sum_correct/sum_total)
    print(f"epoch:{epoch+1} train mean loss={mean_loss}, accuracy={accuracy}")
    train_loss_value.append(mean_loss)
    train_acc_value.append(accuracy)

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    #test dataを使ってテストをする
    for (inputs, labels) in tqdm(testloader, desc=f"epoch:{epoch+1} testing", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    
    mean_loss = sum_loss*BATCH_SIZE/len(testloader.dataset)
    accuracy = float(sum_correct/sum_total)
    print(f"epoch:{epoch+1} test  mean loss={mean_loss}, accuracy={accuracy}")
    test_loss_value.append(mean_loss)
    test_acc_value.append(accuracy)


plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")