import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
import scipy
from pprint import pprint

from datasets.gan_examples import GanExamples


def quantitize_laplace(x, mean, scale, bits=256):

    cdf = np.zeros_like(x)
    for t in range(len(x)):
        if x[t] <= mean:
            cdf[t] = 0.5 * np.exp((x[t] - mean) / scale)
        else:
            cdf[t] = 1. - 0.5 * np.exp(-(x[t] - mean) / scale)

    cdf = np.clip(cdf, 1e-5, 1 - 1e-5)

    q = (cdf * 256 + 0.5).astype(int)

    q = np.clip(q, 1, 254)
    q = q.astype(float) / 256.

    return q


def mu_compand(x, mean, scale, mu=255):

    f_cont = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
