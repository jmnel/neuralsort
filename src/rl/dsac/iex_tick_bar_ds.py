import sqlite3
from pprint import pprint
from datetime import datetime

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd

import matplotlib as mpl
mpl.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import mplfinance as mpf

import settings

RL_PATH = settings.DATA_DIRECTORY / (settings.RL_DATABSE_NAME)


class TickBarDatasetIEX(Dataset):

    def __init__(self, mode='train'):
        super(Dataset, self).__init__()

        db = sqlite3.connect(RL_PATH)

        rows = db.execute('SELECT day, symbol FROM ticks_20_meta;').fetchall()
        self.indices = tuple((r[0], r[1]) for r in rows)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

#        print(f'{idx} of {len(self)}')

        day, symbol = self.indices[idx]

#        print(f'day: {day}, sym: {symbol}')

        db = sqlite3.connect(RL_PATH)

        ticks = db.execute('''
SELECT timestamp, open, high, low, close, volume FROM ticks_20
WHERE day=? AND symbol=?;''', (day, symbol)).fetchall()

        try:
            ticks = torch.FloatTensor(ticks)
        except:
            return self.__getitem__(idx + 1)

        return day, symbol, ticks
#        pprint(type(ticks))
#        print(ticks)
#        ticks = torch.from_numpy(ticks)
#        print(ticks.shape)
#        return ticks


#dl = torch.utils.data.DataLoader(TickBarDatasetIEX())
#it = iter(dl)

# for idx in range(10000):

#    try:
#        res = next(it)
#    except StopIteration:
#        break
#    except TypeError:
#        continue

#    day, symbol, ticks = res

#    ts = ticks[0, :, 0].numpy()
#    o = ticks[0, :, 1]
#    h = ticks[0, :, 2]
#    l = ticks[0, :, 3]
#    c = ticks[0, :, 4]
#    v = ticks[0, :, 5]


#    ts = tuple(datetime.fromtimestamp(int(t)) for t in ts)
#    data = {'Open': o,
#            'High': h,
#            'Low': l,
#            'Close': c,
#            'Volume': v}
#    df = pd.DataFrame(data, index=ts)

#    mpf.plot(df,
#             type='candle',
#             volume=True,
#             tight_layout=True,
#             style='yahoo',
#             title=f'{symbol} - {day}')
