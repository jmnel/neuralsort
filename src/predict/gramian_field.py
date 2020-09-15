from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def transform(x):

    scale_min = x.min()
    scale_max = x.max()
    x_scaled = (2 * x - scale_max - scale_min) / (scale_max - scale_min)

    # Clip to (-1, 1).
    x_scaled = np.clip(x_scaled, -1, 1)

    # Encoder to polar coods.
    phi = np.arccos(x_scaled)
    r = np.linspace(0, 1, len(x_scaled))

    gaf = np.vectorize(lambda a, b: np.cos(a + b))(*np.meshgrid(phi, phi, sparse=True))

    return gaf, phi, r, x_scaled


class Gram(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device

    def forward(self, x):

        scale_min, _ = torch.min(x, dim=-1)
        scale_max, _ = torch.max(x, dim=-1)

        x_scaled = (2 * x - scale_max - scale_min) / (scale_max - scale_min)
        x_scaled = torch.clamp(x_scaled, -1, 1)

        phi = torch.acos(x_scaled)
        g = torch.zeros((x.shape[0], 1, phi.shape[-1], phi.shape[-1])).to(self.device)

        print(f'phi: {phi.shape}')

        for batch_idx in range(x.shape[0]):
            a = phi[batch_idx].repeat(512, 1)
            b = phi[batch_idx].transpose(1, 0)
            b = b.repeat(1, 512)
            g[batch_idx, 0] = torch.cos(a + b)

#        print(g[0].shape)

        return g
