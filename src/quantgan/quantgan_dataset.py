import sqlite3
from random import shuffle
from random import randint
import random
from datetime import datetime
import math
import json

import numpy as np
from torch.utils.data import Dataset
import torch

import settings

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn


class QuantGanDataset(Dataset):

    def __init__(self, window_lenth=100):

        super().__init__()

        self.window_lenth = window_lenth

        # Get list of tickers and associated meta data from database.
        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            rows = db.execute('''
SELECT symbol, meta_json FROM quantgan_meta;
''').fetchall()

            self.symbols_list = list()
            self.meta_info = dict()

            for row in rows:

                symbol, json_meta = row

                self.symbols_list.append(symbol)
                json_meta = json.loads(json_meta)
                self.meta_info[symbol] = json_meta

        # Generate rolling window lookup table.
        lookup_table = list()

        for sym in self.symbols_list:

            num_days = self.meta_info[sym]['num_days']
            for offset in range(0, num_days - window_lenth + 1):
                assert offset + window_lenth <= num_days
                lookup_table.append((sym, self.meta_info[sym], offset))

        self.lookup_table = lookup_table

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        else:
            idx = [idx, ]

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            x_list = list()
            symbols_list = list()
            meta_list = list()

            for i in idx:

                sym, meta, offset = self.lookup_table[i]

                rows = db.execute('''
SELECT x_norm2 FROM quantgan_data WHERE symbol == ? ORDER BY date;
''', (sym,)).fetchall()

                rows = rows[offset:offset + self.window_lenth]
                assert len(rows) == self.window_lenth

                rows = list(r[0] for r in rows)

                x_list.append(torch.FloatTensor(
                    rows).reshape(1, self.window_lenth))
                meta_list.append(self.meta_info[sym])
                symbols_list.append(sym)

        x = torch.cat(x_list, axis=1)

        return x, meta_list, symbols_list

    def __len__(self):
        return len(self.lookup_table)

    class SmallView(Dataset):

        def __init__(self, parent):

            self.lookup_table = parent.lookup_table.copy()
            shuffle(self.lookup_table)

        def __len__(self):
            pass
