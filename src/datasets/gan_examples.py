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

# random.seed(992312)


class GanExamples:

    def __init__(self):

        super().__init__()

        self.start_date = datetime(2001, 1, 1)
        self.end_date = datetime(2019, 12, 1)

        self.start_date_str = self.start_date.strftime('%Y-%m-%d')
        self.end_date_str = self.end_date.strftime('%Y-%m-%d')

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            rows = db.execute('''
SELECT id, symbol, start_date, end_date
FROM qdl_symbols
WHERE start_date <= ? AND end_date >= ?;
            ''', (self.start_date_str, self.end_date_str)).fetchall()

        shuffle(rows)

        sids, symbols, start_dates, end_dates = tuple(zip(*rows))

        start_dates = [datetime.strptime(
            d.split(' ')[0], '%Y-%m-%d').date() for d in start_dates]
        end_dates = [datetime.strptime(
            d.split(' ')[0], '%Y-%m-%d').date() for d in end_dates]

        self.sids = sids
        self.symbols = symbols
        self.start_dates = start_dates
        self.end_dates = end_dates

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        else:
            idx = [idx, ]

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            q = list()
            p = list()

            for i in idx:

                rows = db.execute('''
SELECT adj_close FROM qdl_eod_symbols_view
WHERE symbol == ? AND date >= ? AND date <= ?;
''', (self.symbols[i], self.start_date_str, self.end_date_str)).fetchall()

                if len(rows) < 4097:
                    nxt = (idx[-1] + 1) % self.__len__()
                    return self.__getitem__(nxt)

                rows = rows[len(rows) - 4097:]

                ts = np.array([r[-1] for r in rows])
                x = np.diff(np.log(ts))

#                x = np.log(x / x[0])

#                return x
                x = torch.FloatTensor(x)

                q.append(x)
                p.append(ts)

            return (torch.cat(q, axis=0), np.concatenate(p, axis=0))

    def time_domain(self):
        """
        Gets the dates which form the time domain for the datset's datpoints.
        """

        return self.t_dom
