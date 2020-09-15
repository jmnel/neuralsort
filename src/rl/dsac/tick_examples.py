import sqlite3
from pprint import pprint
from datetime import datetime

import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import settings

IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_SMALL_NAME


class TickExamples(Dataset):

    def __init__(self,
                 mode='train',
                 num_examples=70000):
        super().__init__()

        self.mode = mode
        self.num_examples = num_examples

        self.split = (0.8, 0.1, 0.1)

        db = sqlite3.connect(IEX_PATH)

#        rows = db.execute('''
# SELECT day, symbol, message_count FROM iex_ticks_meta
# WHERE message_count >= 2000;''').fetchall()
        rows = db.execute('''
SELECT day, symbol, message_count FROM iex_meta_2000;''').fetchall()
#        random.shuffle(rows)
        rows = rows[:num_examples]

        n = len(rows)

        self.offsets = (0, int(self.split[0] * n + 0.5), int((self.split[0] + self.split[1]) * n + 0.5))

        self.indices = {'train': rows[self.offsets[0]:self.offsets[1]],
                        'validate': rows[self.offsets[1]:self.offsets[2]],
                        'test': rows[self.offsets[2]:]}

        self.data = {'train': list(), 'validate': list(), 'test': list()}

        max_seq_len = -1

        for mode in ('train', 'validate', 'test'):
            print(f'Loading data for {mode}')
            for idx, (day, symbol, _) in enumerate(self.indices[mode]):

                if idx % 100 == 0:
                    print(f'Loading data: {idx+1} of {len(self.indices[mode])}')

                rows = db.execute('''
SELECT timestamp, price, size FROM iex_trade_reports_2000
WHERE day=? AND symbol=?
ORDER BY timestamp;''', (day, symbol)).fetchall()

                # Rescale timestamp between 0 and 1 for 9:30 -> 16:00.
                d0 = datetime.fromtimestamp(rows[0][0] * 1e-9)
                d1 = d0
                t0 = d0.replace(hour=9, minute=30, second=0, microsecond=0).timestamp()
                t1 = d1.replace(hour=4 + 12, minute=0, second=0, microsecond=0).timestamp()
                rows = list(((r[0] * 1e-9 - t0) / (t1 - t0), *r[1:]) for r in rows)

                max_seq_len = max(len(rows), max_seq_len)

                for k in range(len(rows)):
                    if rows[k][1] == 0:
                        assert k != 0 and k + 1 != len(rows)
                        m = 0.5 * rows[k - 1][1] + 0.5 * rows[k + 1]
                        rows[k] = (rows[k][0], m, rows[k][2])

                self.data[mode].append((day, symbol, len(rows), rows))

#                print(f'{day} -> {symbol}')

        self.max_seq_len = max_seq_len

    def set_mode(self, mode: str):
        if mode not in {'train', 'validate', 'test'}:
            raise ValueError('invalid dataset mode')

        self.mode = mode

    def __len__(self):
        return len(self.indices[self.mode])

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = self.data[self.mode][idx]

        labels = batch[0:3]
        x = torch.FloatTensor(batch[3])

        padding = torch.zeros((self.max_seq_len - x.shape[0], 3))
        mask = torch.cat((torch.ones(x.shape[0]), torch.zeros(padding.shape[0])))
        x = torch.cat((x, padding), dim=0)

        return labels, x, mask


# dl = DataLoader(TickExamples(mode='train'),
#                batch_size=2,
#                shuffle=False)

#_, x, mask = next(iter(dl))

# pprint(x.shape)
# print(mask.shape)
