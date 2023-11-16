import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.net import Net
from torchvision import transforms, datasets


criterion = torch.nn.CrossEntropyLoss()


def train(model: Net, optimizer: Adam, device: torch.device, train_loader: DataLoader, num_epochs: int, progress_bar: bool = True):
    model.train()
   
    for epoch in range(num_epochs):
        for (inputs, labels) in (tqdm(train_loader, desc=f"epoch:{epoch+1} training", leave=False) if progress_bar else train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(model: Net, device: torch.device, test_loader: DataLoader, progress_bar: bool = True):
    sum_correct = 0
    model.eval()

    for (inputs, labels) in (tqdm(test_loader, desc=f"testing", leave=False) if progress_bar else test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        sum_correct += (predicted == labels).sum().item()
    
    accuracy = float(sum_correct/len(test_loader.dataset))
    return accuracy
    


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5), 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])
trainset = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform_train)

transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5,)])
testset = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform_test)