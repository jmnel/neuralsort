import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[1]))

from random import randint
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np
from pprint import pprint

from db_connectors import SQLite3Connector


class WindowedDataset(Dataset):

    def __init__(self,
                 data_path: Path,
                 train_size,
                 test_size,
                 prediction_window,
                 num_stocks,
                 is_train,
                 transform=None):

        super().__init__()

        self._transform = transform

        db = SQLite3Connector.connect(data_path / 'clean.db')

        table = 'adj_returns_clean'

        # Get list of symbols by picking first (n=num_stocks) column names.
        schema = db.get_schema(table)
        symbols = [s['name'] for s in schema[1:]][0:num_stocks]

        # Get actual price time series.
        raw = db.select(table, symbols)
        db.close()

        k_folds = 4
        fold_len = len(raw) // k_folds

        print(len(raw))
        print(fold_len)
#        print(fold_len *


data_path = Path(__file__).absolute().parents[3] / 'data'
print(data_path)
foo = WindowedDataset(data_path,
                      train_size=600,
                      test_size=200,
                      prediction_window=10,
                      num_stocks=5,
                      is_train=True)
