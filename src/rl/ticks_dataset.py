import sqlite3
from pprint import pprint

import torch
import numpy as np
from torch.utils.data import Dataset

import settings

IB_PATH = settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME
IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

print(IEX_PATH)


class TicksDataset(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        db = sqlite3.connect(IB_PATH)

        days = tuple(zip(*db.execute('SELECT date FROM ib_days;').fetchall()))[0]
        days = days[1:2]

        self.data = list()

        size_max = db.execute('''
SELECT MAX(size) FROM ib_trade_reports;
''').fetchall()[0][0]

        for day in days:

            rows = db.execute('''
SELECT count(id), symbol FROM ib_trade_reports WHERE day=? GROUP BY symbol;''', (day,)).fetchall()

            rows = tuple(filter(lambda x: x[0] > 300, rows))

            offset = int(0.8 * len(rows))
            if mode == 'train':
                symbols = tuple(r[1] for r in rows[:offset])
            else:
                symbols = tuple(r[1] for r in rows[offset:])

            rows = db.execute('''
SELECT MIN(timestamp), MAX(timestamp) FROM ib_trade_reports WHERE day=?;
''', (day,)).fetchall()[0]

            tmin, tmax = rows

            for symbol in symbols:
                rows = db.execute('''
SELECT timestamp, price, size FROM ib_trade_reports
WHERE symbol=? AND day=? ORDER BY timestamp''', (symbol, day)).fetchall()

                ts, price, size = zip(*rows)
                ts = tuple((t - tmin) / (tmax - tmin) for t in ts)
                price = tuple(p * 1e-2 for p in price)
                size = tuple(s * 1e-3 for s in size)

                self.data.append((day, symbol, torch.FloatTensor(tuple(zip(ts, price, size)))))

        db.close()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def __len__(self):
        return len(self.data)


class TicksDatasetIEX(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        db = sqlite3.connect(IEX_PATH)

        days = tuple(zip(*db.execute('SELECT date FROM iex_days;').fetchall()))[0]
        days = days[0:4]

        self.data = list()

#        size_max = db.execute('''
# SELECT MAX(size) FROM iex_trade_reports;
# ''').fetchall()[0][0]

        for day in days:

            print(f'Counting ticks for day {day}')
            rows = db.execute('''
SELECT count(id), symbol FROM iex_trade_reports WHERE day=? GROUP BY symbol;''', (day,)).fetchall()

            rows = tuple(filter(lambda x: x[0] > 300, rows))

            offset = int(0.8 * len(rows))
            for r in rows:
                if r[0] > 2000:
                    self.data.append((day, r[1]))
                    print(f'\t{r[1]} -> {r[0]}')
#            if mode == 'train':
#                symbols = tuple(r[1] for r in rows[:offset])
#            else:
#                symbols = tuple(r[1] for r in rows[offset:])

#            rows = db.execute('''
# SELECT MIN(timestamp), MAX(timestamp) FROM ib_trade_reports WHERE day=?;
# ''', (day,)).fetchall()[0]

#            tmin, tmax = rows

#            for symbol in symbols:
#                rows = db.execute('''
# SELECT timestamp, price, size FROM ib_trade_reports
# WHERE symbol=? AND day=? ORDER BY timestamp''', (symbol, day)).fetchall()

#                ts, price, size = zip(*rows)
#                ts = tuple((t - tmin) / (tmax - tmin) for t in ts)
#                price = tuple(p * 1e-2 for p in price)
#                size = tuple(s * 1e-3 for s in size)

#                self.data.append((day, symbol, torch.FloatTensor(tuple(zip(ts, price, size)))))

        db.close()

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        day, symbol = self.data[idx]

        print(f'Getting {day} -> {symbol}')

        db = sqlite3.connect(IEX_PATH)

        rows = db.execute('''
SELECT timestamp, price, size FROM iex_trade_reports WHERE day=? AND symbol=?;
''', (day, symbol)).fetchall()

        ts, price, size = tuple(zip(*rows))
        tmin, tmax = min(ts), max(ts)

        ts = tuple((t - tmin) / (tmax - tmin) for t in ts)
        size = tuple(s * 1e-3 for s in size)

        db.close()

        return day, symbol, torch.FloatTensor(tuple(zip(ts, price, size)))

    def __len__(self):
        return len(self.data)


#q = TicksDatasetIEX()

#day, sym, x = next(iter(q))

#print(f'day: {day}, sym: {sym}')

# pprint(x)
