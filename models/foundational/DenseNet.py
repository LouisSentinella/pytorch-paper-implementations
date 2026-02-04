import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim


class DenseNetBlock(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        out = x
        return out

class DenseNet121(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):


        return x


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    print('cuda:', cuda)

    device = torch.device("cuda" if cuda else "cpu")

    net = DenseNet121()

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

