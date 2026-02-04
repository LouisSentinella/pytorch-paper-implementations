import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim


class ResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block 1
        self.ResNetBlock1 = ResNetBlock(64, 64, stride=1)
        self.ResNetBlock2 = ResNetBlock(64, 64, stride=1)

        # Block 2
        self.ResNetBlock3 = ResNetBlock(64, 128, stride=2)
        self.ResNetBlock4 = ResNetBlock(128, 128, stride=1)

        # Block 3
        self.ResNetBlock5 = ResNetBlock(128, 256, stride=2)
        self.ResNetBlock6 = ResNetBlock(256, 256, stride=1)

        # Block 4
        self.ResNetBlock7 = ResNetBlock(256, 512, stride=2)
        self.ResNetBlock8 = ResNetBlock(512, 512, stride=1)

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512,10)


    def forward(self,x):

        x = self.conv1(x)
        x = self.MaxPool(x)

        x = self.ResNetBlock1(x)
        x = self.ResNetBlock2(x)

        x = self.ResNetBlock3(x)
        x = self.ResNetBlock4(x)

        x = self.ResNetBlock5(x)
        x = self.ResNetBlock6(x)

        x = self.ResNetBlock7(x)
        x = self.ResNetBlock8(x)

        x = self.GlobalAvgPool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    print('cuda:', cuda)

    device = torch.device("cuda" if cuda else "cpu")

    net = ResNet18()

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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epochs = 1

    for epoch in range(epochs):
        net.train()
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

    net.eval()
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total
    print("Test Accuracy: {:.4f}".format(accuracy))

