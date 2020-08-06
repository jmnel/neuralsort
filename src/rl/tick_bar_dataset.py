import sqlite3
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset

import settings

IB_PATH = settings.DATA_DIRECTORY / (settings.IB_DATABASE_NAME + '-2')
print(IB_PATH)


class TickBarDataset(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        db = sqlite3.connect(IB_PATH)

        days = tuple(zip(
            *db.execute('SELECT DISTINCT(day) FROM ticks_5_meta;').fetchall()))[0]

#        days = days[-1:]

        self.data = list()

        for day in days:
            rows = db.execute('SELECT symbol, count FROM ticks_5_meta WHERE day=? ORDER BY count;',
                              (day,)).fetchall()

            rows = list(filter(lambda x: x[1] >= 400, rows))

            for symbol, _ in rows:
                ticks = db.execute('''
SELECT idx, timestamp, open, high, low, close, volume FROM ticks_5
WHERE day=? AND symbol=? ORDER BY idx;''', (day, symbol)).fetchall()

                ticks = list((t[0] / ticks[-1][0],
                              (t[1] - ticks[0][1]) / (ticks[-1][1] - ticks[0][1]),
                              *t[2:]) for t in ticks)
                ticks = torch.FloatTensor(ticks)
                ticks = ticks.reshape((ticks.shape[0], ticks.shape[1]))

                self.data.append((day, symbol, ticks))

        db.close()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        day, symbol, ticks = self.data[idx]
        ticks = torch.FloatTensor(ticks)

        return day, symbol, ticks.reshape((ticks.shape[0], 7))

    def __len__(self):
        return len(self.data)
