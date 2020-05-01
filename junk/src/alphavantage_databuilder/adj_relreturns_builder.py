import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

from db_connectors import SQLite3Connector
from datetime import datetime
from pprint import pprint
import sqlite3

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import numpy as np


class AdjRelReturnsBuilder:

    def __init__(self,
                 data_path: Path):

        av_db_path = data_path / 'av.db'
        av_db = SQLite3Connector.connect(av_db_path)

        dates = av_db.select('daily_adjusted', ['date'])
        dates = list([datetime.strptime(d[0], '%Y-%m-%d') for d in dates])

        filter_cls = 'WHERE ts_type == "daily_adj"'
        symbols = av_db.select('symbols_meta', ['symbol'], filter_cls)
        symbols = {s[0] for s in symbols}
        symbols -= {'DOW', 'UHN', 'V'}

        min_dates = list()
        max_dates = list()
        for s in symbols:
            dmin = av_db.select('daily_adjusted', ['min(date)'],
                                'WHERE symbol=="{}"'.format(s))
            dmax = av_db.select('daily_adjusted', ['max(date)'],
                                'WHERE symbol=="{}"'.format(s))

            dmin = datetime.strptime(dmin[0][0], '%Y-%m-%d')
            dmax = datetime.strptime(dmax[0][0], '%Y-%m-%d')

            min_dates.append((dmin, s))
            max_dates.append((dmax, s))

        min_date = max(min_dates)
        max_date = min(max_dates)

#        print(min_date)
#        print(max_date)

        assert(all(min_dates[0][0] == x[0] for x in min_dates))
        assert(all(max_dates[0][0] == x[0] for x in max_dates))

        min_date = datetime.strftime(min_date[0], '%Y-%m-%d')
        max_date = datetime.strftime(max_date[0], '%Y-%m-%d')

        data = dict()
        n = len(symbols)
        m = 0
        for s in symbols:
            filter_cls = f'WHERE'
            filter_cls += f' symbol == "{s}"'
            filter_cls += f' and date >= "{min_date}"'
            filter_cls += f' and date <= "{max_date}"'
            close_price = av_db.select('daily_adjusted', ['adj_close'],
                                       filter_cls)

            data[s] = list([float(x[0]) for x in close_price])
            m = len(data[s]) - 1
#            m = len(data[s])

        # Done with source database.
        av_db.close()

        rel_returns = np.zeros((m, n))
        for j, s in enumerate(symbols):
            for i in range(m):
                r0 = data[s][i]
                r1 = data[s][i + 1]
                rr = np.clip((r1 - r0) / r1, -4.0, 0.5)
                rr = np.power(np.abs((r1 - r0) / r0), 1 / 8)
                rel_returns[i, j] = rr
#                rel_returns[i, j] = data[s][i]

        # Open 'clean.db' sqlite3 database; create if doesn't exist'
        db_clean = SQLite3Connector.connect(data_path / 'clean.db')

        # Delete relative returns table if it exsists.
        try:
            db_clean.drop_table('relative_returns_clean')

        except sqlite3.OperationalError as e:
            print('WARNING: Table \'relative_returns_clean\' doesn\'t exist.')

        # Create table 'relative_returns'.
        cols_props = [
            {'name': 'id', 'dtype': 'INTEGER', 'not_null': True, 'pk': True}
        ]
        for j, s in enumerate(symbols):
            cols_props.append({'name': s, 'dtype': 'FLOAT'})

        db_clean.create_table('relative_returns_clean', cols_props)

        values = [[None, ] + rel_returns[i].tolist()
                  for i in range(len(rel_returns))]

        cols = ['id', ] + [s for s in symbols]

        db_clean.insert('relative_returns_clean', cols, values)
        db_clean.commit()
        db_clean.close()


data_path = Path(__file__).absolute().parent.parent.parent / 'data'
print(data_path)
builder = AdjRelReturnsBuilder(data_path)
