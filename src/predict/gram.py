from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import settings
from rl.dsac.tick_examples import TickExamples
import gramian_field as gf

DEVICE = torch.device('cuda')
NUM_LAYERS = 5


class Encoder(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.gram = gf.Gram(device)

        self.conv_in = nn.Conv2d(1, 2**NUM_LAYERS, 8, 1)
        self.layers = nn.ModuleList()
        final_size = 512 - 8 + 1
        for layer_idx in range(NUM_LAYERS):
            i = NUM_LAYERS - layer_idx
            self.layers.append(nn.Conv2d(2**i, 2**(i - 1), 8, 1))
            final_size = (final_size - 8 + 1)
            if layer_idx + 1 < NUM_LAYERS:
                final_size = final_size // 2

        out_size = final_size ** 2

        self.fc_1 = nn.Linear(out_size, 64)
        self.fc_2 = nn.Linear(64, 64)

        self.to(device)

    def forward(self, x_input):
        x = self.gram(x_input)

        x = self.conv_in(x)

        for layer_idx in range(NUM_LAYERS):
            x = self.layers[layer_idx](x)
            x = F.relu(x)
            if layer_idx + 1 < NUM_LAYERS:
                x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)

        print(x.shape)

        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

        self.layers = nn.ModuleList()
        final_size = 512 - 8 + 1
        for layer_idx in range(NUM_LAYERS):
            i = layer_idx
            self.layers.append(nn.ConvTranspose2d(2**i, 2**(i + 1), 8, 1))

        self.fc_1 = nn.Linear(64, 64)
        self.fc_2 = nn.Linear(64, 324)

        self.dconv_out = nn.ConvTranspose2d(2**NUM_LAYERS, 1, 8)

        self.to(device)

    def forward(self, x_in):

        x = self.fc_1(x_in)
        x = F.relu(x)
        x = self.fc_2(x)

        for layer_idx in range(NUM_LAYERS):

        print(x.shape)
        exit()


class Model(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

        self.encoder = Encoder(device)
        self.decoder = Decoder(device)

        self.to(device)

    def forward(self, x_in):
        x = self.encoder(x_in)
        x = self.decoder(x)

        return x


loader = DataLoader(dataset=TickExamples(num_examples=10),
                    batch_size=1,
                    shuffle=True,
                    drop_last=True)

labels, ticks, mask = next(iter(loader))
seq_len = labels[2][0]

price = ticks[0, :seq_len, 1:2]

log_price = np.log(price)

x = log_price
# scaler = MinMaxScaler(feature_range=(-1, 1))
# x = scaler.fit_transform(log_price)

x = x.reshape((1, 1, x.shape[0]))
x = x[:, :, :512]

x = torch.FloatTensor(x).to(DEVICE)
model = Model(DEVICE)
y = model(x)

print(y.shape)
# y = gaf(x)

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(y[0].cpu().numpy())
# axes[1].plot(np.arange(x.shape[2]), x[0, 0, :].cpu())
# print(x.shape)

# print(y[0].shape)
# axes[1].imshow(gaf)
# plt.show()
