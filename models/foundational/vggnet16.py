import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim


class VGGNet(nn.Module):
    """
    VVGGNet model

    This is modified to have 10 output channel for CIFAR10, as I cannot fit ImageNet or anything larger on my laptop.
    """

    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.5)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)


    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    print('cuda:', cuda)

    device = torch.device("cuda" if cuda else "cpu")

    net = VGGNet()

    net.to(device)

    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=True,
                                           transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../../datasets', train=False,
                                                transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

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

