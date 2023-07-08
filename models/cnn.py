import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pathlib import Path


import sys;

sys.path.insert(0, '..')  # Enable import from parent folder.
from utils import plot_model_performance


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class ReluNet(nn.Module):
    def __init__(self):
        super(ReluNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1600, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x

    def forward_threshold(self, x, threshold=1.0):
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, 2)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        loss = loss.item()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % 10 == 0:
            print('\r[{}/{} ({:.0f}%)]\tLoss = {:.4f}\tAccuracy = {:.2f}'.format(
                (batch_idx + 1) * len(data), len(train_loader.dataset), 100 * batch_idx / len(train_loader),
                loss / len(target), 100 * correct / len(target)), end='')
        total_loss += loss
        total_correct += correct
    mean_loss = total_loss / len(train_loader.dataset)
    acc = total_correct / len(train_loader.dataset)
    print("\r" + 100 * " ", end='')
    print('\rEpoch {} \t Loss = {:.4f}\t Accuracy = {:.2f}%'.format(epoch, mean_loss, float(100 * acc)), end='')
    return mean_loss, acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    print('Loss = {:.4f}\tAccuracy = {:.2f}%'.format(test_loss, 100. * correct / len(test_loader.dataset)))
    return test_loss, acc


def main(debug=False):
    path = Path("mnist/cnn_debug" if debug else "mnist/cnn")
    epochs = 5 if debug else 30
    lr_decay = .9
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)
    if debug:
        train_set = torch.utils.data.Subset(train_set, np.random.choice(np.arange(len(train_set)), 500))
        test_set = torch.utils.data.Subset(test_set, np.random.choice(np.arange(len(train_set)), 500))
    train_set = torch.utils.data.DataLoader(train_set, batch_size=512)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=512)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters())

    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)
    print(f"Training Model for {epochs} epochs.", flush=True)
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(1, epochs + 1):
        loss, acc = train(model, device, train_set, optimizer, epoch)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        print("\t\tTest Set:\t", end="")
        loss, acc = test(model, device, test_set)
        history["val_loss"].append(loss)
        history["val_accuracy"].append(acc)
        scheduler.step()
    print("\nTrain Set\t", end="")
    loss, acc = test(model, device, train_set)
    print("Test Set\t", end="")
    loss_t, acc_t = test(model, device, test_set)
    path.mkdir(parents=True, exist_ok=True)
    plot_model_performance(history, path / "training.png")
    torch.save(model, path / "model.pt")
    with open(path / "performance.txt", 'w') as file:
        print(f"Train: \t Loss: \t {loss} \t Accuacy: \t {acc}", file=file)
        print(f"Test: \t Loss: \t {loss_t} \t Accuacy: \t {acc_t}", file=file)
    print()


if __name__ == '__main__':
    main(debug=False)
