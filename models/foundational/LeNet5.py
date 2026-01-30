import numpy as np
import torchvision
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch import nn, optim


class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)

if __name__ == '__main__':
    net = LeNet5()

    dataset = torchvision.datasets.MNIST(root='../../datasets', train=True, transform=T.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../datasets', train=False, transform=T.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(dataloader):

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print("Epoch[{}/{}], Step[{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, 100, batch_idx + 1, len(dataloader), loss.item()))

    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = net(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = correct / total
    print("Test Accuracy: {:.4f}".format(accuracy))

