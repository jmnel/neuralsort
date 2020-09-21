import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader


class SeBlock2d(nn.Module):

    def __init__(self, channels, ratio=2):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=channels,
                              channels,
                              kernel_size=3,
                              padding=1)

        self.gate_1 = nn.Linear(in_features=channels,
                                out_features=channels // ratio)
        self.gate_2 = nn.Linear(in_features=channels // ratio,
                                out_features=channels)

    def forward(self, x):

        u = self.conv(x)
        h, w = x.shape[-2:]

        z = 1. / (h + w) * torch.sum(u, dim=(-2, -1))

        s = torch.sigmoid(self.gate_2(F.relu(self.gate_1(z))))
        s = s.unsqueeze(-1).unsqueeze(-1)

#        print(f's={s.shape}')
#        print(f'u={u.shape}')

        x_out = s * u
#        print(f'out={x_out.shape}')

        return x_out
