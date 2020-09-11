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
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima import auto_arima

import settings

from rl.dsac.tick_examples import TickExamples

NUM_EXAMPLES = 10
PRED_LEN = 200

loader = DataLoader(TickExamples(mode='train', num_examples=NUM_EXAMPLES),
                    batch_size=1,
                    shuffle=True)

labels, ticks, mask = next(iter(loader))
seq_len = labels[2][0]

price = ticks[0, :seq_len, 1].numpy()

price = np.log(price)

sc_in = MinMaxScaler(feature_range=(0, 1))

price = price.reshape(price.shape[0], 1)
price = sc_in.fit_transform(price)
#price = price.flatten()

# mu = np.mean(price)
# price -= mu
# std = np.std(price)
# price /= std

# log_returns = np.diff(np.log(price))

TEST_SIZE = 1500

x_train = price[:-TEST_SIZE - 1, :]
y_train = price[1:-TEST_SIZE, :]
x_test = price[-TEST_SIZE - 1:, :]
y_test = price[-TEST_SIZE - 1:, :]

# print(x_test.shape)
# print(y_test.shape)

#assert len(x_test) + len(x_train) == seq_len
# exit()

# x = log_returns[:seq_len - 1 - PRED_LEN]
# y_hat = log_returns[seq_len - 1 - PRED_LEN:]

# x = price[:-PRED_LEN]
# y_hat = price[-PRED_LEN:]

# step_wise = auto_arima(y_train,
#                       exogenous=x_train,
#                       start_p=1, start_q=1,
#                       max_p=7, max_q=7,
#                       d=1, max_d=7,
#                       trace=True,
#                       error_action='ignore',
#                       supress_warnings=True,
#                       stepwise=True)

x_train = x_train.flatten()
y_train = y_train.flatten()
x_test = x_test.flatten()
y_test = y_test.flatten()
model = SARIMAX(y_train,
                exog=x_train,
                order=(1, 0, 3),
                enforce_invertibility=False,
                enforce_stationarity=False)


fit = model.fit()

forecast = fit.forecast(steps=TEST_SIZE, exog=x_test[:-1])

plt.plot(np.arange(len(y_train)), y_train)
plt.plot(np.arange(len(y_test)) + len(y_train), y_test)
plt.plot(np.arange(len(y_test) - 1) + len(y_train), forecast, linewidth=2.0)
pprint(forecast)
plt.show()

# d = 1
# p = 1
# q = 10

# model = ARIMA(x, order=(d, p, q))
# model = SARIMAX(x, order=(d, p, q))

# model_fit = model.fit()

# y = model_fit.forecast(PRED_LEN, alpha=0.05)

# pprint(y)
# exit()

# plt.plot(np.arange(len(x)), x, linewidth=0.4)
# plt.plot(np.arange(len(y_hat)) + len(x), y_hat, linewidth=0.4)
# plt.plot(np.arange(len(y_hat)) + len(x), y, linewidth=0.4)

# print(len(y))

# plt.show()

# print(model_fit.summary())


# plt.show()


# seq_len = labels[2].item()
# price = price[:seq_len]
# PREDICT_LENGTH = 100

# plt.plot(price)
# plt.show()


# def optimize_params(trial: optuna.trial.Trial):
#    return {
#        'order_p': trial.suggest_int('order_p', 0, 10),
#        'order_d': trial.suggest_int('order_d', 0, 10),
#        'order_q': trial.suggest_int('order_q', 0, 10),
#        'seasonal_p': trial.suggest_int('seasonal_p', 0, 50),
#        'seasonal_d': trial.suggest_int('seasonal_d', 0, 3),
#        'seasonal_q': trial.suggest_int('seasonal_q', 1, 50),
#        'seasonal_s': trial.suggest_int('seasonal_s', 0, 50),
#    }


# def fit(hyper_params):
#    loader.dataset.set_mode('train')
#    order = (hyper_params['order_p'], hyper_params['order_d'], hyper_params['order_q'])
#    seasonal_order = (hyper_params['seasonal_p'],
#                      hyper_params['seasonal_d'],
#                      hyper_params['seasonal_q'],
#                      hyper_params['seasonal_s'])
#    loss = 0
#    for idx, (labels, ticks, mask) in enumerate(loader):
#        print(f'Train: {idx+1} of {len(loader)}')
#        seq_len = labels[2].item()
#        price = ticks[0, :seq_len, 1]
#        price = price.numpy()
#        x = price[:-PREDICT_LENGTH]
#        y_hat = price[-PREDICT_LENGTH:]
#        model = ARIMA(endog=x, order=order)
#        model_fit = model.fit()
#        y = model_fit.forecast(steps=PREDICT_LENGTH)
#        loss += np.mean((y - y_hat)**2)

#        if idx == 0:
#            plt.plot(np.arange(len(x)), x)
#            plt.plot(np.arange(len(y_hat)) + len(x), y_hat)
#            plt.plot(np.arange(len(y)) + len(x), y)
#            plt.show()

#    print(f'loss: {loss}')
#    sys.stdout.flush()
#    return loss


# def validate():
#    pass


# def test():
#    pass


# def obj(trial):
#    params = optimize_params(trial)
#    loss = fit(params)

#    return loss


# def optimize(n_trials=10, n_jobs=15):
#    study = optuna.create_study(study_name='test', load_if_exists=False)
#    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs)


# optimize()

# study = optuna.load_study(study_name='test', storage='sqlite:///params.db')
# params = study.best_trial.params

# pprint(params)
