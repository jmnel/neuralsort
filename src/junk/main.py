#!/usr/bin/env python3

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline

trian_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, pin_memory=True)

STITCH_SIZE = 4
stitch = torch.FloatTensor()
stitch_label = 0
for i in range(STITCH_SIZE):
    while True:
        data, label = next(iter(trian_loader))
        if label != 0:
            break
    stitch = torch.cat([stitch, data[0][0]], 1)
    stitch_label += label*10**(STITCH_SIZE - i - 1)

plt.imshow(stitch)
print(stitch_label)
