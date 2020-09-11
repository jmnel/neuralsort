from pprint import pprint

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import seaborn as sb

from iex_ticks_dataset import TickDatasetIEX

train_loader = DataLoader(TickDatasetIEX(mode='train'),
                          batch_size=1,
                          shuffle=True)

train_iter = iter(train_loader)

for idx, (day, symbol, x) in enumerate(train_iter):
    p = x[0, :, 1]
    log_r = np.diff(np.log(p)) * 1e2
