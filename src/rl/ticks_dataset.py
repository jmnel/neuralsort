import sqlite3
from pprint import pprint

import numpy as np
from torch.utils.data import Dataset

import settings

IB_PATH = settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME


class TicksDataset(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        db = sqlite3.connect(IB_PATH)

        days = tuple(zip(*db.execute('SELECT date FROM ib_days;').fetchall()))[0]

        data = list()

        for day in days:

            rows = db.execute('''
SELECT count(id), symbol FROM ib_trade_reports WHERE day=? GROUP BY symbol;''', (day,)).fetchall()

            rows = tuple(filter(lambda x: x[0] > 300, rows))

            pprint(rows)

            offset = int(0.8 * len(rows))
            if mode == 'train':
                symbols = tuple(r[1] for r in rows[:offset])
            else:
                symbols = tuple(r[1] for r in rows[offset:])

            for symbol in symbols:
                rows = db.execute('''
SELECT timestamp, price, size FROM ib_trade_reports
WHERE symbol=? AND day=? ORDER BY timestamp''', (symbol, day)).fetchall()

        db.close()

    def __getitem__(idx):
        pass

    def __len__():
        pass


q = TicksDataset()
