import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))
import random
from datetime import datetime
import pandas as pd
from pprint import pprint
import numpy as np

random.seed(1)

from db_connectors import SQLite3Connector

data_path = Path(__file__).absolute().parents[2] / 'data'

db = SQLite3Connector.connect(data_path / 'eoddata.db')

# Set minimum date range.
min_date_range = ('2000-01-01', '2020-01-06')

# Get list of all symbols in database.
symbols_min = db.select('eoddata_raw',
                        ['symbol'], f'WHERE date <= "{min_date_range[0]}"')
symbols_max = db.select('eoddata_raw',
                        ['symbol'], f'WHERE date >= "{min_date_range[1]}"')
symbols_min = set(row[0] for row in symbols_min)
symbols_max = set(row[0] for row in symbols_max)


where_cls = 'GROUP by symbol'
symbol_counts = db.select('eoddata_raw',
                          ['symbol, count(symbol)'], where_cls)
counts = (sc[1] for sc in symbol_counts)
max_count = max(counts)

symbols_with_max = filter(lambda sc: sc[1] == max_count, symbol_counts)
symbols_with_max = (sc[0] for sc in symbols_with_max)


symbols = list(symbols_min.intersection(symbols_max, symbols_with_max))


random.shuffle(symbols)

n = 10

symbols = symbols[: n]

try:
    db.drop_table('ohlcv100')
except:
    pass


where_cls = ', '.join(f'"{s}"' for s in symbols)
where_cls = 'WHERE symbol in ( ' + where_cls + ' )'

raw = db.select('eoddata_raw',
                ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
                where_cls)

sym_min_dates = {s: datetime(2100, 1, 1) for s in symbols}
sym_max_dates = {s: datetime(1900, 1, 1) for s in symbols}
for i in range(len(raw)):
    date, sym, open, high, low, close, volume = raw[i]
    date = datetime.strptime(date, '%Y-%m-%d')

    sym_min_dates[sym] = min(sym_min_dates[sym], date)
    sym_max_dates[sym] = max(sym_max_dates[sym], date)

for s in symbols:
    print(f'{s}  :  {sym_min_dates[s]}  ->  {sym_max_dates[s]}')

min_date = max(sym_min_dates.values())
max_date = min(sym_max_dates.values())

where_cls += f' AND date >= "{min_date}" and date <= "{max_date}"'
raw = db.select('eoddata_raw',
                ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'],
                where_cls)

df = dict()


# pprint(raw)

for row in raw:

    date_str, sym, open, high, low, close, volume = row
    date = datetime.strptime(date_str, '%Y-%m-%d')
    if date >= min_date and date <= max_date:

        if not date_str in df:
            df[date_str] = dict()

        df[date_str]['date'] = date_str
        df[date_str][f'{sym}_open'] = open
        df[date_str][f'{sym}_high'] = high
        df[date_str][f'{sym}_low'] = low
        df[date_str][f'{sym}_close'] = close
        df[date_str][f'{sym}_volume'] = volume


cols = [{'name': 'id', 'dtype': 'INTEGER', 'pk': True, 'not_null': True}]
cols += [{'name': 'date', 'dtype': 'DATE'}]

for s in symbols:
    cols.append({'name': f'{s}_open', 'dtype': 'FLOAT'})
    cols.append({'name': f'{s}_high', 'dtype': 'FLOAT'})
    cols.append({'name': f'{s}_low', 'dtype': 'FLOAT'})
    cols.append({'name': f'{s}_close', 'dtype': 'FLOAT'})
    cols.append({'name': f'{s}_volume', 'dtype': 'FLOAT'})

cols_names = list(c['name'] for c in cols[1:])

df = pd.DataFrame.from_dict(df, orient='index', columns=cols_names)
df = df.sort_index()

vals = df.values

m = df.shape[0]
idx_col = np.array([None for _ in range(m)]).reshape((m, 1))
vals = np.concatenate((idx_col, df.values), axis=1)

try:
    db.drop_table('ohlcv100')
except:
    pass

db.create_table('ohlcv100', cols)

db.insert('ohlcv100', cols, vals)

db.commit()

db.close()
