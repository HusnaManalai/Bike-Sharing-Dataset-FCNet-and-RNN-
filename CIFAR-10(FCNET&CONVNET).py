import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Creating data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Fully Connected Neural Network for CIFAR-10
class CIFAR_FCNet(nn.Module):
    def __init__(self):
        super(CIFAR_FCNet, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # Flatten images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

# Convolutional Neural Network for CIFAR-10
class CIFAR_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)  # 3 input channels (RGB), 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
model_fc = CIFAR_FCNet().to(device)
model_conv = CIFAR_ConvNet().to(device)

# Optimizers
optimizer_fc = optim.SGD(model_fc.parameters(), lr=0.01, momentum=0.5)
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.5)

# Training and testing functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Training and testing the models on CIFAR-10
for epoch in range(1, 11):
    train(model_fc, device, train_loader, optimizer_fc, epoch)
    test(model_fc, device, test_loader)
    train(model_conv, device, train_loader, optimizer_conv, epoch)
    test(model_conv, device, test_loader)
