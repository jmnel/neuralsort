import sqlite3
from random import shuffle, randint
import random
from datetime import datetime
import math
import json
from pprint import pprint
import math

import numpy as np
from torch.utils.data import Dataset
import torch

import settings


class DatasetView(Dataset):

    def __init__(self,
                 parent,
                 mode):

        self.parent = parent
        self.mode = mode

        self.num_stocks = parent.num_stocks
        self.lookup_table = parent.lookup_table[mode]
        self.symbols_list = parent.symbols_list

    def __len__(self):
        return len(symbols_list) * len(lookup_table)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        else:
            idx = [idx, ]

        # Map index to stock and offset rolling window.
        for i in idx:
            i_stock = i // self.num_stocks
            i_window = i % self.num_stocks


class WaveNetNsortDataset():

    def __init__(self,
                 window_lenth=100,
                 split_ratio=(7, 2, 1),
                 num_stocks=100):

        super().__init__()

        self.window_lenth = window_lenth
        self.split_ratio = split_ratio
        self.num_stocks = num_stocks

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

#            pprint(self.meta_info[symbol])

        # Generate train, validate, test splits.
        num_days = self.meta_info[self.symbols_list[0]]['num_days']

        portion_train = split_ratio[0] / sum(split_ratio)
        portion_validate = split_ratio[1] / sum(split_ratio)
        portion_test = split_ratio[2] / sum(split_ratio)

        offset_train = 0
        offset_validate = math.floor(portion_train * num_days)
        offset_test = offset_validate + math.floor(portion_test * num_days)

        ranges = {'train': (0, offset_validate),
                  'validate': (offset_validate, offset_test),
                  'test': (offset_test, num_days)}

        # Take subset of stocks to decrease dataset size.
        assert num_stocks <= len(self.symbols_list)
        shuffle(self.symbols_list)
        self.symbols_list = self.symbols_list[:num_stocks]

        # Generate rolling window lookup table.
        lookup_table = dict()
        for mode in ranges.keys():

            lookup_table[mode] = list()
            t_start, t_end = ranges[mode]
            for idx in range(t_start, t_end - window_lenth + 1):
                assert idx + window_lenth <= num_days
                assert idx + window_lenth <= t_end
                lookup_table[mode].append(idx)

        self.lookup_table = lookup_table


foo = WaveNetNsortDataset()
