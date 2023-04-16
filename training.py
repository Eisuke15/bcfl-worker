import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from net import Net


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
    
    