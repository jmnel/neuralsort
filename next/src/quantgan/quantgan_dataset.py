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


class QuantGanDataset:

    def __init__(self):

        super().__init__()

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

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        else:
            idx = [idx, ]

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

            x = list()
            meta = list()

            for i in idx:
                symbol = self.symbols_list[i]
                rows = db.execute('''
SELECT x_norm2 FROM quantgan_data WHERE symbol == ? ORDER BY date;
''', (symbol,)).fetchall()

                rows = rows[-4096:]

                rows = list(row[0] for row in rows)

                x.append(torch.FloatTensor(rows).reshape(1, 4096))
                meta.append(self.meta_info[symbol])

            x = torch.cat(x, axis=1)

            return x, meta, tuple(self.symbols_list[i] for i in idx)
