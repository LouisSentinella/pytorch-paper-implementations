import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim


class AlexNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    print('cuda:', cuda)

    device = torch.device("cuda" if cuda else "cpu")

    net = AlexNet()

    net.to(device)

    transform = T.Compose([
        T.Resize(227),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=True,
                                           transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=False,
                                                transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epochs = 10

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print("Epoch[{}/{}], Step[{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, epochs, batch_idx + 1, len(dataloader), loss.item()))
        scheduler.step()

    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total
    print("Test Accuracy: {:.4f}".format(accuracy))

