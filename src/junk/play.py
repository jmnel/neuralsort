import torch
import numpy as np
import math
from torchvision import datasets, transforms
from mnist_stitched_dataset import StitchedMNIST
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(41472, 5184)
        self.fc2 = nn.Linear(5184, 648)
        self.fc3 = nn.Linear(648, 162)
        self.fc4 = nn.Linear(162, 40)
#        self.fc2 = nn.Linear(10368, 5832)
#        self.fc2 = nn.Linear(128, 10)
#        self.fc2 = nn.Linear( , 5832

        # 28 / 26 / 24 / 12 * 64
        # 112 / 110 / 108 / 54 * 64

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = F.log_softmax(x, dim=1)
#        x = self.dropout2(x)
#        x = self.fc2(x)

        return x
#        output = F.log_softmax(x, dim=1)
#        return output


train_loader = torch.utils.data.DataLoader(
    StitchedMNIST('../data/mnist_stitched.pkl',
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(
                          mean=(0.1307, ), std=(0.3081, ))
                  ])),
    batch_size=1, shuffle=True, pin_memory=True)


model = CnnModel()

data, target = next(iter(train_loader))


# plt.imshow(data[0][0])
# plt.show()

output = model(data)
print(output)
#        print(f'target=\n{target}')
#        loss = F.nll_loss(output, target)
#loss = F.cross_entropy(output, target)
