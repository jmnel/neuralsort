from datetime import datetime
from pathlib import Path
from pprint import pprint
import random
from random import shuffle
import json
import math
import operator
import sqlite3
import torch

import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import settings


class RelativeReturnDataset(Dataset):

    def __init__(self,
                 num_samples: int,
                 num_stocks: int,
                 mode: str,
                 forecast_window: int = 20,
                 transform=None,
                 split_ratio=(6, 2, 2),
                 use_cache: bool = False,
                 number_of_days: int = 4000):

        super().__init__()

        self.num_samples = num_samples
        self.num_stocks = num_stocks
        self.mode = mode
        self.forecast_window = forecast_window
        self.transform = transform
        self.split_ratio = split_ratio
        self.use_cache = use_cache

        if mode not in ('train', 'validate', 'test'):
            raise ValueError('mode must be one of train, validate, test')

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
            stock_list, stock_meta = self.get_list_of_stocks(db)
            ts = build_timeseries(db, stock_list, stock_meta)

    def get_list_of_stocks(self, db: sqlite3.Connection):

        rows = db.execute('''
SELECT id, symbol, start_date, end_date, lifetime_returns FROM qdl_symbols;
''').fetchall()

        date_fmt = '%Y-%m-%d'
        stock_list_all = shuffle([row[1] for row in rows])

        for row in rows:
            stock_id = row[0]
            symbol = row[1]
            print(symbol)
            start_date = row[2]
            end_date = row[3]
            life_ret = row[4]

            assert(stock_id is not None)
            assert(symbol is not None)
            assert(start_date is not None)
            assert(end_date is not None)
            assert(life_ret is not None)

        exit()

        stock_meta_all = {row[1]: (row[0],
                                   datetime.strptime(row[2].split(
                                       ' ')[0], date_fmt).date(),
                                   datetime.strptime(row[3].split(
                                       ' ')[0], date_fmt).date(),
                                   row[4])
                          for row in rows}

        stock_list = list()
        stock_meta = dict()
        i_pick = 0
        while len(stock_list < num_stocks):
            stock = stock_list_all[i_pick]
            i_pick += 1
            meta = stock_meta[stock]

            if meta[1] < datetime(2000, 1, 1) and meta[2] >= datetime(2020, 4, 17):
                stock_list.append(stock)
                stock_meta[stock] = meta
                print(f'added {stock}')

        return stock_list, stock_meta

    def build_timeseries(self, db: sqlite3.Connection, stock_list, stock_meta):

        # Get latest common starting date.
        start_date = min(meta[1] for meta in stock_meta.values())
        print(f'start : {start_date}')


#        for symbol, properties in stock_list.items():
#            print(symbol)

test = RelativeReturnDataset(num_samples=200,
                             num_stocks=10,
                             mode='train',
                             forecast_window=100)
