import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        return out


class GoogLeNet(nn.Module):
    """
    GoogLeNet model

    This is modified to have 10 output channel for CIFAR10, as I cannot fit ImageNet or anything larger on my laptop.
    """
    def __init__(self):
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True))
        self.Pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))
        self.Pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.Inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.Pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.Inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.Inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.Inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.Inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.Pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.Inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        # self.Pool5 = nn.MaxPool2d(kernel_size=7, stride=1)
        self.Pool5 = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(in_features=1024, out_features=10)


    def forward(self,x):
        x = self.Conv1(x)
        x = self.Pool1(x)
        x = self.Conv2(x)
        x = self.Pool2(x)
        x = self.Inception3a(x)
        x = self.Inception3b(x)
        x = self.Pool3(x)
        x = self.Inception4a(x)
        x = self.Inception4b(x)
        x = self.Inception4c(x)
        x = self.Inception4d(x)
        x = self.Inception4e(x)
        x = self.Pool4(x)
        x = self.Inception5a(x)
        x = self.Inception5b(x)
        x = self.Pool5(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    print('cuda:', cuda)

    device = torch.device("cuda" if cuda else "cpu")

    net = GoogLeNet()

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
    epochs = 1

    net.train()

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

