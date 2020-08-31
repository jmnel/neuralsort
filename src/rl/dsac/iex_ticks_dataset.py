import sqlite3
from pprint import pprint

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import settings

IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME


class TickDatasetIEX(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        db = sqlite3.connect(IEX_PATH)

        rows = db.execute('SELECT day, symbol, message_count FROM iex_ticks_meta WHERE message_count >= 2000;').fetchall()
        rows = rows[:300]
#        rows = list(filter(lambda r: r[2] >= 5000, rows))

#        random.shuffle(rows)
        print(f'Initializing with {len(rows)} examples.')

        self.index = list(r[:-1] for r in rows)
        self.data = list()
        for idx, (day, symbol, _) in enumerate(rows):

            if idx % 100 == 0:
                print(f' {idx + 1} of {len(rows)}')

            rows2 = db.execute('''
    SELECT timestamp, price, size FROM iex_trade_reports WHERE day=? AND symbol=?;
    ''', (day, symbol)).fetchall()

            if len(rows2) <= 2:
                continue

#            a = rows2[-1][0] - rows2[0][0]
#            n = len(rows2)
#            print(f'a: {a}, n: {n}')

            rows2 = list(((r[0] - rows2[0][0]) / (rows2[-1][0] - rows2[0][0]), *r[1:]) for r in rows2)
            rows2 = torch.FloatTensor(rows2)
            self.data.append((day, symbol, rows2))

        db.close()

        print(f'Dataset initialized with {len(self.data)} examples.')

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        day, symbol, rows = self.data[idx]

#        day, symbol = self.index[idx]

#        db = sqlite3.connect(IEX_PATH)

#        rows = db.execute('''
# SELECT timestamp, price, size FROM iex_trade_reports WHERE day=? AND symbol=? ORDER BY timestamp;
# ''', (day, symbol)).fetchall()

#        rows = list(((r[0] - rows[0][0]) / (rows[-1][0] - rows[0][0]), *r[1:]) for r in rows)
#        rows = torch.FloatTensor(rows)
#        rows = np.array(rows, dtype=float)
#        rows[:, 0] = (rows[:, 0] - rows[0, 0]) / (rows[-1, 0] - rows[0, 0])

#        db.close()

        return day, symbol, rows.reshape((rows.shape[0], 3))

    def __len__(self):
        return len(self.data)


#q = DataLoader(TickDatasetIEX())
