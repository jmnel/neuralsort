import sqlite3
from pprint import pprint
from random import shuffle
from random import randint
from datetime import datetime

import numpy as np
from torch.utils.data import Dataset

import settings


class StockBasket:

    saved_state = None

    def __init__(self,
                 num_stocks,
                 start_date=datetime(2000, 1, 1),
                 end_date=datetime(2020, 1, 1)):

        if not StockBasket.saved_state:

            StockBasket.saved_state = {'num_stocks': num_stocks,
                                       'start_date': start_date,
                                       'end_date': end_date}

            self.num_stocks = num_stocks
            self.start_date = start_date
            self.end_date = end_date

            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            # Get list of tickers within date range and associated meta data.
            with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
                rows = db.execute(f'''
    SELECT symbol, start_date, end_date, lifetime_returns
    FROM qdl_symbols
    WHERE start_date < '{start_date_str}' AND end_date > '{end_date_str}'
    ORDER BY lifetime_returns DESC;
                ''').fetchall()

            # Parse date column.
            rows = [(r[0],
                     datetime.strptime(r[1].split(' ')[0], '%Y-%m-%d').date(),
                     datetime.strptime(r[2].split(' ')[0], '%Y-%m-%d').date(),
                     r[3]) for r in rows]

            # Parition list of stocks into 10 subsets.
            partition_k = 10
            offsets = [i * len(rows) //
                       partition_k for i in range(partition_k + 1)]

            partition = [rows[offsets[i]:offsets[i + 1]]
                         for i in range(partition_k)]

            stock_list = list()

            # Randomly pick stocks from each subset of partition in round-robbin fashion.
            i = 0
            while len(stock_list) < num_stocks:
                i = i % partition_k

                i_pick = randint(0, len(partition[i]) - 1)
                stock_list.append((*partition[i][i_pick], i))

                i += 1

            self.stock_list = stock_list

            StockBasket.saved_state['stock_list'] = stock_list

        else:

            self.num_stocks = StockBasket.saved_state['num_stocks']
            self.start_date = StockBasket.saved_state['start_date']
            self.end_date = StockBasket.saved_state['end_date']
            self.stock_list = StockBasket.saved_state['stock_list']

            assert(self.num_stocks == num_stocks)
            assert(self.start_date == start_date)
            assert(self.end_date == self.end_date)
            assert(self.stock_list == self.stock_list)

    def get_list(self):
        return self.stock_list
