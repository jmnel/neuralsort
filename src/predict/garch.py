import sys
from pprint import pprint

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

import settings

from rl.dsac.tick_examples import TickExamples

NUM_EXAMPLES = 10

loader = DataLoader(TickExamples(mode='train', num_examples=NUM_EXAMPLES),
                    batch_size=1,
                    shuffle=True)

labels, ticks, mask = next(iter(loader))

price = ticks[0, :, 1]
seq_len = labels[2].item()
price = price[:seq_len]
PREDICT_LENGTH = 100

log_returns = np.diff(np.log(price))

x = log_returns[:-PREDICT_LENGTH]

SCALE = 1e3

log_returns *= SCALE

am = arch_model(log_returns, vol='GARCH', p=5, o=2, q=5, dist='Normal')
res = am.fit(update_freq=1)

y = res.forecast(horizon=5)
y_hat = log_returns[-PREDICT_LENGTH:]

#assert len(y) == len(y_hat)

mu = y.mean.iloc[-1].values
plt.plot(mu)
#plt.plot(np.arrange(len(x)), x)
#plt.plot(np.arrange(len(y)) + len(x), y)
#plt.plot(np.arrange(len(y)) + len(x), y_hat)
plt.show()
