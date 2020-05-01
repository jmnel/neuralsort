from datetime import datetime
from pathlib import Path
from pprint import pprint
import random
from random import randint
import json
import math
import operator
import sqlite3
import torch

import pickle
import numpy as np
from torch.utils.data import Dataset
import quandl
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import settings

# random.seed(0)

# test_symbols = ['MSFT', 'TSM', 'PG', 'GS',
#                'HD', 'JNJ', 'JPM', 'UNH', 'KO', 'PFE', 'RGEN', 'RL', 'BAK', 'MGA', 'NLSN', 'CNST', 'FOXF', 'SIGI', 'ROAD', 'CAAS', 'STAG', 'OI', 'HURN', 'HEI', 'GHM', 'CMCSA', 'NTGR', 'FMC', 'RTX']
date_cutoff = datetime(year=2020, month=1, day=15).date()
# date_cutoff = datetime(year=2022, month=1, day=15).date()


class WindowNorm(Dataset):

    def __init__(self,
                 num_samples: int,
                 mode: str,
                 num_stocks: int,
                 num_attributes: int = 5,
                 forecast_window: int = 20,
                 transform=None,
                 split_ratio=(6, 2, 2),
                 use_cache: bool = False):

        super().__init__()

        self.num_samples = num_samples
        self.mode = mode
        self.num_stocks = num_stocks
        self.num_attributes = num_attributes
        self.forecast_window = forecast_window
        self.split_ratio = split_ratio

        self.transform = transform

        if mode not in {'train', 'validate', 'test'}:
            raise ValueError('mode must be one of {train, validate, test}')

        if use_cache:
            cache_rebuild = False
            cache_path = settings.DATA_DIRECTORY / f'window_norm-{mode}.pkl'

            if cache_path.is_file():
                with open(cache_path, 'rb') as cache_file:
                    self.data = pickle.load(cache_file)
                    return

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            available_tickers = [row[0] for row in db.execute('''
SELECT symbol FROM qdl_symbols
WHERE start_date < '2001-06-01';
''').fetchall()]

            random.shuffle(available_tickers)

            available_tickers = available_tickers[:num_stocks]

#            available_tickers = {ticker: list()
#                                 for ticker in available_tickers}

            sql = '''
SELECT * FROM qdl_eod_symbols_view WHERE symbol IN({});
'''.format(','.join('?' for _ in available_tickers))

            data = db.execute(sql, available_tickers).fetchall()

        ts = {ticker: list() for ticker in available_tickers}

        for row in data:
            ticker, date, o, h, l, c, v = row
            date = datetime.strptime(date, '%Y-%m-%d').date()

            ts[ticker].append((date, o, h, l, c, v))

        # Apply cutoff and clip to maximum available date range.
        start_date = datetime(1900, 12, 1).date()
        end_date = datetime(2100, 12, 1).date()

        for ticker, rows in ts.items():
            ts[ticker] = list(filter(lambda r: r[0] <= date_cutoff,
                                     sorted(rows, key=operator.itemgetter(0))))

        for ticker, rows in ts.items():
            start_date = max(rows[0][0], start_date)
            end_date = min(rows[-1][0], end_date)

        for ticker in ts.keys():
            ts[ticker] = list(filter(
                lambda r: r[0] >= start_date and r[0] <= end_date, ts[ticker]))

            dp_count = {ticker: 0 for ticker in ts.keys()}

        num_timesteps = len(ts[available_tickers[0]])
        for ticker in dp_count.keys():
            dp_count[ticker] = len(ts[ticker])
            print(f'{ticker} : {dp_count[ticker]} {num_timesteps}')
            assert(dp_count[ticker] == num_timesteps)

        num_attributes = 5

        data = np.zeros((num_timesteps, num_stocks, num_attributes))

        for t in range(num_timesteps):
            for idx, ticker in enumerate(available_tickers):
                for jdx in range(num_attributes):
                    #                    print(ticker)
                    #                    data[t, idx, jdx] = 0
                    #                    print(f'({t}, {idx}, {jdx})')
                    #                    print(f'{len(data)}')
                    #                    print(f'{len(ts[ticker])}')
                    data[t, idx, jdx] = ts[ticker][t][jdx + 1]

        train_portion = split_ratio[0] / sum(split_ratio)
        validate_portion = split_ratio[1] / sum(split_ratio)
        test_portion = split_ratio[2] / sum(split_ratio)

        validate_start = math.floor(train_portion * num_timesteps + 0.5)
        test_start = math.floor(
            (train_portion + validate_portion) * num_timesteps + 0.5)

        if mode == 'train':
            data = data[:validate_start]
        elif mode == 'validate':
            data = data[validate_start:test_start]
        else:
            data = data[validate_start:]

        self.data = list()

        for i in range(num_samples):

            #            print(len(data))
            seq_start = randint(0, len(data) - 1 - forecast_window - 2 - 6)
            assert(seq_start + forecast_window + 6 < len(data))

            # Normalize open, high, low, and close.
            t0 = seq_start
            t1 = seq_start + forecast_window
#            t2 = seq_start + forecast_window + 7

            window = np.array(data[t0:t1], copy=True)

            assert(len(window) == forecast_window)

            stds = np.zeros(num_stocks)
            for j in range(num_stocks):

                #                window[:, j,
                #                high_mu = np.mean(window[:, j, 1])
                #                low_mu = np.mean(window[:, j, 2])
                #                mu = 0.5 * (low_mu + high_mu)
                #                high_lows = np.concatenate((window[:, j, 2],
                #                                            window[:, j, 1]),
                #                                           axis=0)

                #                stds[j] = np.std(high_lows)

                #                window[:, j, :] -= mu

                #                window[:, j, :] = np.log(window[:, j, :])

                o_mag = window[0, j, 0]
                h_mag = window[0, j, 1]
                l_mag = window[0, j, 2]
                c_mag = window[0, j, 3]
                v_mag = window[0, j, 0]
                window[:, j, 0:4] -= window[0, j, 0:4]
                window[:, j, 0] /= o_mag
                window[:, j, 1] /= h_mag
                window[:, j, 2] /= l_mag
                window[:, j, 3] /= c_mag
                window[:, j, 4] = 0.0

                #            avg_std = np.mean(stds)

                #            for j in range(num_stocks):
                #                window[:, j, 0:4] /= avg_std

                # Normalize volume.
                #            vols = window[:, :, -1]
                #            vols_mu = np.mean(vols)
                #            vols_std = np.std(vols)
                #            window[:, :, -1] -= vols_mu
                #            window[:, :, -1] /= vols_std

            seq_input = torch.FloatTensor(
                window)

            t_next_day = seq_start + forecast_window + 1
            pred_open = data[t_next_day, :, 0]
            pred_close = data[t_next_day + 7, :, 3]
            rel_returns = (pred_close - pred_open) / pred_open
#            rel_returns -= np.mean(rel_returns)
#            rel_returns /= np.std(rel_returns)
            seq_label = torch.FloatTensor(rel_returns)

            seq_label = torch.FloatTensor(data[t_next_day])

            self.data.append((seq_input, seq_label, data[t_next_day]))

        if use_cache:
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(self.data, cache_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_input, seq_label, actual_next_day = self.data[idx]

        if self.transform:
            seq_input = self.transform(seq_input)
            seq_label = self.transform(seq_label)

        return (seq_input, seq_label, actual_next_day)


# foo = WindowNorm(num_samples=600,
#                 num_stocks=10,
#                 forecast_window=200,
#                 mode='train')

#x, y, _ = next(iter(foo))

# for j in range(foo.num_stocks):
#    plt.plot(x[:, j, 0])

# plt.show()
