import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCnn2(nn.Module):
    def __init__(self, num_digits):
        super(DeepCnn2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5,
                               stride=1)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=5,
                               stride=1)

        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        print(f'before pool1 = {x.shape}')
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


x = torch.ones((3, 1, 28, 3 * 28))

model = DeepCnn2(3)

x = model(x)

print(x.shape)

for name, param in model.named_parameters():
    #        if param.requires_grad:
    if True:
        print(f'{name} : {param.data.shape}')
#        print(model.parameters)
