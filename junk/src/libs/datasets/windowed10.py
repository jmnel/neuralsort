import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

from random import randint, shuffle
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pprint import pprint
import math
import itertools

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import pickle

from db_connectors import SQLite3Connector

_windows10_fold_perm = None


class Windowed10Dataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 num_samples=600,
                 forecast_window=20,
                 num_stocks=3,
                 train=True,
                 transform=None,
                 cache_enabled=True):

        super().__init__()

        self.transform = transform

        refresh_cache = True
        if cache_enabled:
            if train:
                cache_path = data_path / 'cache' / 'windowed10_cache-train.pkl'
            else:
                cache_path = data_path / 'cache' / 'windowed10_cache-test.pkl'

            if cache_path.is_file():
                with open(cache_path, 'rb') as file:
                    cached = pickle.load(file)
                if (cached['num_samples'] == num_samples and
                    cached['forecast_window'] == forecast_window and
                        cached['num_stocks'] == num_stocks):
                    self.data = cached['data']
                    refresh_cache = False
                    return

        db = SQLite3Connector.connect(data_path / 'eoddata.db')
        schema = db.get_schema('ohlcv100')
        columns = tuple(s['name'] for s in schema)

        raw = np.array(db.select('ohlcv100'))
        raw = np.delete(raw, [0, 1], axis=1)

        idx_dicard = list(i for i in range(num_stocks * 5, raw.shape[1]))
        raw = np.delete(raw, idx_dicard, axis=1)

        columns = columns[2:]
        columns = columns[:5 * num_stocks]

        df = pd.DataFrame.from_records(raw, columns=columns)
        df = df.astype(float)

        k_fold = 10

        fold_len = len(raw) // k_fold

        assert(forecast_window < fold_len)

        fold_offsets = list(fold_len * i for i in range(k_fold))

        global _windows10_fold_perm
        if _windows10_fold_perm is None:
            _windows10_fold_perm = list(i for i in range(k_fold))
            shuffle(_windows10_fold_perm)

        fold_perms = _windows10_fold_perm
        fold_offsets = list(fold_offsets[p] for p in fold_perms)

        for fo in fold_offsets:
            assert(fo + fold_len <= raw.shape[0])

        k_fold_train = math.floor((k_fold / 10) * 8)
        k_fold_test = k_fold - k_fold_train

        fold_offsets_train = fold_offsets[:k_fold_train]
        fold_offsets_test = fold_offsets[-k_fold_test:]

        fold_offsets = fold_offsets_train if train else fold_offsets_test

        self.data = list()

        for i in range(num_samples):

            # Pick random fold train/test set.
            fold_start = fold_offsets[randint(0, len(fold_offsets) - 1)]
            fold_end = fold_start + fold_len

            assert(fold_end - fold_start == fold_len)

            # Pick random subset of fold.
            seq_start = randint(fold_start, fold_end - 1 - forecast_window - 1)

            assert(seq_start + forecast_window <= fold_end)

            # Normalize open, high, low, and close.
            t0 = seq_start
            t1 = seq_start + forecast_window + 1
            stds = np.zeros(num_stocks)
            for j in range(num_stocks):

                high_mu = np.mean(df.iloc[t0:t1, 5 * j].values)
                low_mu = np.mean(df.iloc[t0:t1, 1 + 5 * j].values)
                mu = 0.5 * (low_mu + high_mu)
                high_low_vals = np.concatenate([df.iloc[t0:t1, 5 * j].values,
                                                df.iloc[t0:t1, 1 + 5 * j].values], axis=0)
                stds[j] = np.std(high_low_vals)

                df.iloc[t0:t1, j * 5:(j + 1) * 5 - 1] -= mu

            avg_std = np.mean(stds)

            for j in range(num_stocks):
                df.iloc[t0:t1, j * 5:(j + 1) * 5 - 1] /= avg_std

            # Normalize volume.
            vols = df.iloc[t0:t1, 4::5].values
            vols_mu = np.mean(vols)
            vols_std = np.std(vols)
            df.iloc[t0:t1, 4::5] -= vols_mu
            df.iloc[t0:t1, 4::5] /= vols_std

            seq_input = torch.FloatTensor(
                df.iloc[seq_start:seq_start + forecast_window].values)

            seq_label = torch.FloatTensor(
                df.iloc[seq_start + forecast_window].values)

            self.data.append((seq_input, seq_label))

        if cache_enabled and refresh_cache:
            if train:
                cache_path = data_path / 'cache' / 'windowed10_cache-train.pkl'
            else:
                cache_path = data_path / 'cache' / 'windowed10_cache-test.pkl'

            cached = {'num_samples': num_samples,
                      'forecast_window': forecast_window,
                      'num_stocks': num_stocks,
                      'data': self.data}

            with open(cache_path, 'wb') as file:
                pickle.dump(cached, file)


#            if cache_path.is_file():
#                cached = pickle.load(cache_path)
#                if (cached['num_samples'] == num_samples and
#                    cached['forecast_window'] == forecast_window and
#                        cached['num_stocks'] == num_stocks):
#                    self.data = cached['data']
#                    return

#            print(seq_input)
#            print(seq_label)

#            x = seq_input
#            open0 = list([x[t][0] for t in range(len(x))])
#            high0 = list([x[t][5] for t in range(len(x))])
#            low0 = list([x[t][10] for t in range(len(x))])
#            close0 = list([x[t][14] for t in range(len(x))])

#            plt.plot(high0, linewidth=0.4, label='high')
#            plt.plot(low0, linewidth=0.4, label='low')
#            plt.plot(open0, linewidth=0.4, label='open')
#            plt.plot(close0, linewidth=0.4, label='close')
#            plt.legend()
#            plt.show()

#            label = df.iloc[seq_start +
#                            forecast_window: seq_start + forecast_window + 1]


#            seq = torch.FloatTensor(seq)
#            label = torch.FloatTensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq_input, seq_label = self.data[idx]

        if self.transform:
            seq_input = self.transform(seq_input)

        return (seq_input, seq_label)


#data_path = Path(__file__).absolute().parents[3] / 'data'
# print(data_path)
#foo = Windowed10Dataset(data_path, forecast_window=500)
# bar = Windows10Dataset(data_path)
