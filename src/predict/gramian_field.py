from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt


def transform(x):

    scale_min = x.min()
    scale_max = x.max()
    x_scaled = (2 * x - scale_max - scale_min) / (scale_max - scale_min)

    # Clip to (-1, 1).
#    x_scaled = np.clip(x_scaled, -1, 1)

    # Encoder to polar coods.
    phi = np.arccos(x_scaled)
    r = np.linspace(0, 1, len(x_scaled))

    gaf = np.vectorize(lambda a, b: np.cos(a - b))(*np.meshgrid(phi, phi, sparse=True))

    return gaf, phi, r, x_scaled


def inverse(x):

    pass


class GASF(torch.autograd.Function):

    #    def __init__(self, device):
    #        super().__init__()

    #        self.device = device

    @staticmethod
    def forward(ctx, x):

        scale_min = torch.min(x)
        scale_max = torch.max(x)

        x_scaled = (x - scale_min) / (scale_max - scale_min)

        x_scaled = torch.clamp(x_scaled, 0, 1)
        phi = torch.acos(x_scaled)

        one_1 = torch.ones(x.shape[0], 1, x.shape[-1]).to(x.device)
        one_2 = torch.transpose(one_1, dim0=-2, dim1=-1)

        a = torch.matmul(one_2, phi)
        b = torch.matmul(torch.transpose(phi, dim0=-2, dim1=-1), one_1)

        g_1 = torch.cos(1.0 * (a + b)).unsqueeze(dim=1)
        g_2 = torch.sin(1.0 * (a - b)).unsqueeze(dim=1)

        return g_1, g_2, scale_min, scale_max


class InverseGASF(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, g, scale_min, scale_max):

        g = torch.clamp(g, -1, 1)

        x = torch.sqrt((torch.diagonal(g[:, 0], dim1=-2, dim2=-1) + 1 + 1e-5) * 0.5)
        x = x * (scale_max - scale_min) + scale_min

        return x

#    @staticmethod
#    def backward(
