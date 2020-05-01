import sqlite3
from pprint import pprint
from random import shuffle
from random import randint
import random
from datetime import datetime
import math

import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import settings
from .basket import StockBasket

# random.seed(992312)


class DatasetView(Dataset):

    def __init__(self, dataset, mode):

        self.dataset = dataset
        self.mode = mode

    def __len__(self):
        return len(self.dataset.data[self.mode])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dataset.data[self.mode][idx]


class TestDataset:

    def __init__(self,
                 num_stocks,
                 split_ratio,
                 forecast_len,
                 hold_len):

        super().__init__()

        self.num_stocks = num_stocks
        self.split_ratio = split_ratio
        self.forecast_len = forecast_len

        basket = StockBasket(num_stocks)

        start_date_str = basket.start_date.strftime('%Y-%m-%d')
        end_date_str = basket.end_date.strftime('%Y-%m-%d')

        tickers, _, _, _, true_overall_rank = zip(*basket.get_list())

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            data = dict()
            for ticker in tickers:
                data[ticker] = db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod_symbols_view
WHERE symbol == ? AND date >= ? AND date <= ?
ORDER BY date;
''', (ticker, start_date_str, end_date_str)).fetchall()

        dates = list()
        for _, rows in data.items():
            for row in rows:
                dates.append(row[0])

        dates = sorted(tuple(set(dates)))
        dates = {d: i for i, d in enumerate(dates)}

        tickers = {t: i for i, t in enumerate(tickers)}

        eod_df = np.ones((len(dates), num_stocks, 5)) * np.NaN

        self.t_dom = [datetime.strptime(d.split(' ')[0], '%Y-%m-%d').date()
                      for d in dates.keys()]

        self.stock_labels = list()

        for ticker, rows in data.items():
            jdx = tickers[ticker]
            self.stock_labels.append(ticker)
            for row in rows:
                idx = dates[row[0]]

                eod_df[idx, jdx, :] = row[1:]

        # Detect and handle missing values.
        missing_values = np.argwhere(np.isnan(eod_df))
        self.num_missing_vals = len(missing_values)

        for t, j, i in missing_values:
            eod_df[t, j, i] = 1.0

        # Double check that all NaN have been filled.
        assert(len(np.argwhere(np.isnan(eod_df))) == 0)

#        missing = np.argwhere(np.isnan(eod_df))
#        if missing.shape[0] > 0:
#            print(missing)
#            exit()

        # Create features.
        features_df = np.copy(eod_df)

        # EMA
        gamma = 2
        l0 = 1
        l1 = 12
        l2 = 26
        l3 = 50
        num_timesteps = len(features_df)

        for j in range(num_stocks):

            for t in range(num_timesteps):

                c = features_df[t, j, 3]
#                features_df[t, j, 0] = c * (gamma / (1 + l0))
                features_df[t, j, 1] = c * (gamma / (1 + l1))
                features_df[t, j, 2] = c * (gamma / (1 + l2))
                features_df[t, j, 3] = c * (gamma / (1 + l3))

                if t == 0:
                    #                    features_df[t, j, 0] = c
                    features_df[t, j, 1] = c
                    features_df[t, j, 2] = c
                    features_df[t, j, 3] = c
                else:
                    #                    features_df[t, j, 0] += features_df[t -
                    #                                                        1, j, 0] * (1 - gamma / (1 + l0))
                    features_df[t, j, 1] += features_df[t -
                                                        1, j, 1] * (1 - gamma / (1 + l1))
                    features_df[t, j, 2] += features_df[t -
                                                        1, j, 2] * (1 - gamma / (1 + l2))
                    features_df[t, j, 3] += features_df[t -
                                                        1, j, 3] * (1 - gamma / (1 + l3))

        # Generate splits.

        portion_dem = sum(split_ratio)
        portions = {'train': split_ratio[0] / portion_dem,
                    'validate': split_ratio[1] / portion_dem,
                    'test': split_ratio[2] / portion_dem
                    }

        offsets = {'train': 0,
                   'validate': math.floor(portions['train'] * num_timesteps + 0.5),
                   'test': math.floor((portions['train'] + portions['validate']) * num_timesteps + 0.5)
                   }

        MODES = ('train', 'validate', 'test')

        self.num_samples = {'train': offsets['validate'] - forecast_len - hold_len,
                            'validate': offsets['test'] - offsets['validate'] - forecast_len - hold_len,
                            'test': num_timesteps - offsets['test'] - forecast_len - hold_len}

        self.data = {m: list() for m in MODES}

        # Construct samples for TRAIN, VALIDATE, and TEST.
        for mode in MODES:

            # The size of the sampling window constrains how many samples we can
            # get. Samples are generated by sliding a window through the particular
            # mode's timeseries.
            for idx in range(self.num_samples[mode]):

                # These are the offsets for the window.
                t0 = idx
                t1 = idx + forecast_len
                t2 = t1 + hold_len

                window = np.copy(features_df[t0:t2])

                for i_stock in range(window.shape[1]):

                    # For now, perform log normalization on the each of the
                    # 5 attributes.
                    #                    window[:, i_stock, 0] = np.log(
                    #                        window[:, i_stock, 0] / window[0, i_stock, 0])
                    #                    window[:, i_stock, 1] = np.log(
                    #                        window[:, i_stock, 1] / window[0, i_stock, 1])
                    #                    window[:, i_stock, 2] = np.log(
                    #                        window[:, i_stock, 2] / window[0, i_stock, 2])
                    #                    window[:, i_stock, 3] = np.log(
                    #                        window[:, i_stock, 3] / window[0, i_stock, 3])

                    # Volume is unused; set it to 0 for now.
                    window[:, i_stock, 4] = 0.0

                # The example is the sequence for the forecasting period.
                seq_x = window[0:forecast_len, :, :]

                # The label is the following timeseries, over which the security
                # is held before being sold.
                seq_y = window[forecast_len:forecast_len + hold_len, :, :]

                # Just predict closing (log-normed) price.
                target = (seq_y[hold_len - 1, :, 0] -
                          seq_y[0, :, 0]) / seq_y[0, :, 0]

                seq_x = torch.FloatTensor(seq_x)
                target = torch.FloatTensor(target)

                self.data[mode].append((seq_x, target))

    def train_view(self, num_samples=None):
        """
        Returns a view on dataset with the train portion.
        """

        return DatasetView(self, 'train')

    def validate_view(self, num_samples=None):
        """
        Returns a view on dataset with the validation portion.
        """

        return DatasetView(self, 'validate')

    def test_view(self, num_samples=None):
        """
        Returns a view on dataset with test portion.
        """

        return DatasetView(self, 'test')

    def stock_labels(self):
        """
        Gets the ticker label for the stocks in dataset.
        """

        return self.stock_labels

    def time_domain(self):
        """
        Gets the dates which form the time domain for the datset's datpoints.
        """

        return self.t_dom
