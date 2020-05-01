import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

from random import randint
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pprint import pprint
import math
import itertools
import pickle

from db_connectors import SQLite3Connector


class Windowed102Dataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 num_samples=600,
                 forecast_window=20,
                 num_stocks=3,
                 mode='train',
                 transform=None,
                 cache_enabled=True,
                 split_ratio=(6, 2, 2)):

        super().__init__()

        self.transform = transform

        assert(mode in {'train', 'validate', 'test'})

        refresh_cache = True
        if cache_enabled:
            if mode == 'train':
                cache_path = data_path / 'cache' / 'windowed102_cache-train.pkl'
            elif mode == 'validate':
                cache_path = data_path / 'cache' / 'windowed102_cache-validate.pkl'
            else:
                cache_path = data_path / 'cache' / 'windowed102_cache-test.pkl'

            if cache_path.is_file():
                with open(cache_path, 'rb') as file:
                    cached = pickle.load(file)
                if (cached['num_samples'] == num_samples and
                        cached['forecast_window'] == forecast_window and
                        cached['num_stocks'] == num_stocks and
                        cached['split_ratio'] == split_ratio):
                    self.data = cached['data']
                    refresh_cache = False
                    return

        db = SQLite3Connector.connect(data_path / 'eoddata.db')
        schema = db.get_schema('ohlcv100')
        columns = tuple(s['name'] for s in schema)

        # Dispose id and date columns.
        raw = np.array(db.select('ohlcv100'))
        raw = np.delete(raw, [0, 1], axis=1)

        # Only keep n stocks; discard the rest.
        idx_dicard = list(i for i in range(num_stocks * 5, raw.shape[1]))
        raw = np.delete(raw, idx_dicard, axis=1)

        # Do the same discard with column names.
        columns = columns[2:]
        columns = columns[:5 * num_stocks]

        # Convert to Pandas dataframe.
        df = pd.DataFrame.from_records(raw, columns=columns)
        df = df.astype(float)

        data_len = len(df)

        # Calculate ratios for train, validate, and test portions.
        train_portion = split_ratio[0] / sum(split_ratio)
        validate_portion = split_ratio[2] / sum(split_ratio)
        test_portion = split_ratio[1] / sum(split_ratio)

        # Get offsets for train, validate, and test portions.
        validate_start = math.floor(train_portion * len(df) + 0.5)
        test_start = math.floor(
            (train_portion + validate_portion) * len(df) + 0.5)

#        train_raw = df[:validate_start]
#        validate_raw = df[validate_start:test_start]
#        test_raw = df[test_start:]

        if mode == 'train':
            df = df[:validate_start]
        elif mode == 'validate':
            df = df[validate_start:test_start]
        else:
            df = df[test_start:]

        self.data = list()

        for i in range(num_samples):

            # Pick a random subset offset.
            seq_start = randint(0, len(df) - 1 - forecast_window - 1)
            assert(seq_start + forecast_window < len(df))

            # Normalize open, high, low, and close.
            t0 = seq_start
            t1 = seq_start + forecast_window + 1
            stds = np.zeros(num_stocks)
            for j in range(num_stocks):
                high_mu = np.mean(df.iloc[t0:t1, 5 * j].values)
                low_mu = np.mean(df.iloc[t0:t1, 1 + 5 * j].values)
                mu = 0.5 * (low_mu + high_mu)
                high_low_vals = np.concatenate([df.iloc[t0:t1, 5 * j].values,
                                                df.iloc[t0:t1, 1 + 5 * j].values],
                                               axis=0)
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
            if mode == 'train':
                cache_path = data_path / 'cache' / 'windowed102_cache-train.pkl'
            elif mode == 'validate':
                cache_path = data_path / 'cache' / 'windowed102_cache-validate.pkl'
            else:
                cache_path = data_path / 'cache' / 'windowed102_cache-test.pkl'

            cached = {'num_samples': num_samples,
                      'forecast_window': forecast_window,
                      'num_stocks': num_stocks,
                      'split_ratio': split_ratio,
                      'data': self.data}

            with open(cache_path, 'wb') as file:
                pickle.dump(cached, file)

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
# foo = Windowed102Dataset(data_path,
#                         num_samples=600,
#                         forecast_window=20,
#                         num_stocks=3,
#                         mode='train',
#                         transform=None,
#                         cache_enabled=False,
#                         split_ratio=(6, 2, 2))
