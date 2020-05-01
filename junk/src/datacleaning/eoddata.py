import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

import glob
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go

from db_connectors import SQLite3Connector


data_path = Path(__file__).absolute().parents[2] / 'data'

build_raw = True

if build_raw:

    db = SQLite3Connector.connect(data_path / 'eoddata.db')

    cols = [{'name': 'id', 'dtype': 'INTEGER', 'not_null': True, 'pk': True},
            {'name': 'symbol', 'dtype': 'CHAR', 'size': 16},
            {'name': 'date', 'dtype': 'DATE'},
            {'name': 'open', 'dtype': 'FLOAT'},
            {'name': 'high', 'dtype': 'FLOAT'},
            {'name': 'low', 'dtype': 'FLOAT'},
            {'name': 'close', 'dtype': 'FLOAT'},
            {'name': 'volume', 'dtype': 'INTEGER'}]

    try:
        db.drop_table('eoddata_raw')
    except:
        print('INFO: Not deleting \'eoddata_raw\'.')

    db.create_table('eoddata_raw', cols)

    csv_paths = list((data_path / 'eoddata' / 'NASDAQ').glob('*.csv'))

    for i, p in enumerate(csv_paths):

        df = pd.read_csv(p)

        v_id = np.array([None for _ in range(len(df))]).reshape((len(df), 1))

        vals = np.concatenate((v_id, df.values), axis=1)

        vals_filtered = list()

        for j in range(len(vals)):

            if vals[j][1].count('.') == 0 and vals[j][1].count('-') == 0:
                vals_filtered.append(vals[j])

        vals = np.array(vals_filtered)

        for j in range(len(vals)):
            d = datetime.strptime(vals[j][2], '%d-%b-%Y')
            vals[j][2] = datetime.strftime(d, '%Y-%m-%d')

        cols = ['id', 'symbol', 'date', 'open',
                'high', 'low', 'close', 'volume']

        db.insert('eoddata_raw', cols, vals)

        if i % 100 == 0:
            print(f'file {i} / {len(csv_paths)} done.')

    db.commit()
    db.close()
    print('done')


# Filter out non common stocks.
#db = SQLite3Connector.connect(data_path / 'eoddata.db')

#db = SQLite3Connector.connect(data_path / 'eoddata.db')

# data = db.select('eoddata_raw', ['symbol', 'date', 'close'],
#                 'WHERE symbol in ("MSFT","AAPL")')

#df = dict()

# for r in data:
#    sym, date, close = r
#    date = datetime.strptime(date, '%Y-%m-%d')
#    date = pd.to_datetime(date)
#    if date in df:
#        df[date][sym] = close
#    else:
#        df[date] = {sym: close}

#s = pd.DataFrame.from_dict(df, orient='index', columns=['MSFT', 'AAPL'])
#s = s.sort_index()

# s.plot(linewidth=0.4)

# plt.show()
