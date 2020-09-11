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

# plt.plot(price)
# plt.show()


def optimize_params(trial: optuna.trial.Trial):
    return {
        'order_p': trial.suggest_int('order_p', 0, 10),
        'order_d': trial.suggest_int('order_d', 0, 10),
        'order_q': trial.suggest_int('order_q', 0, 10),
        #        'seasonal_p': trial.suggest_int('seasonal_p', 0, 50),
        #        'seasonal_d': trial.suggest_int('seasonal_d', 0, 3),
        #        'seasonal_q': trial.suggest_int('seasonal_q', 1, 50),
        #        'seasonal_s': trial.suggest_int('seasonal_s', 0, 50),
    }


def fit(hyper_params):
    loader.dataset.set_mode('train')
    order = (hyper_params['order_p'], hyper_params['order_d'], hyper_params['order_q'])
#    seasonal_order = (hyper_params['seasonal_p'],
#                      hyper_params['seasonal_d'],
#                      hyper_params['seasonal_q'],
#                      hyper_params['seasonal_s'])
    loss = 0
    for idx, (labels, ticks, mask) in enumerate(loader):
        print(f'Train: {idx+1} of {len(loader)}')
        seq_len = labels[2].item()
        price = ticks[0, :seq_len, 1]
        price = price.numpy()
        x = price[:-PREDICT_LENGTH]
#        x = np.diff(np.log(x))
        y_hat = price[-PREDICT_LENGTH:]
        model = ARIMA(endog=x, order=order)
        model_fit = model.fit()
        y = model_fit.forecast(steps=PREDICT_LENGTH)
#        print(y)
        loss += np.mean((y - y_hat)**2)

        if idx == 0:
            plt.plot(np.arange(len(x)), x)
            plt.plot(np.arange(len(y_hat)) + len(x), y_hat)
            plt.plot(np.arange(len(y)) + len(x), y)
            plt.show()

    print(f'loss: {loss}')
    sys.stdout.flush()
    return loss


def validate():
    pass


def test():
    pass


def obj(trial):
    params = optimize_params(trial)
    loss = fit(params)

    return loss


def optimize(n_trials=10, n_jobs=15):
    study = optuna.create_study(study_name='test', load_if_exists=False)
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)


optimize()

# study = optuna.load_study(study_name='test', storage='sqlite:///params.db')
# params = study.best_trial.params

# pprint(params)
