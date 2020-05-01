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
import pandas as pd

import plotly.graph_objs as go

data_path = Path(__file__).absolute().parents[2] / 'data'

# db = SQLite3Connector.connect(data_path / 'av.db')

# conn = sqlite3.connect(data_path / 'av.db')
# sym = 'MSFT'

db = SQLite3Connector.connect(data_path / 'av.db')

raw = db.select('daily', w_filter='where symbol == "MSFT"')
m = len(raw)

raw2 = {'date': [r[2] for r in raw],
        'open': np.array([r[3] for r in raw]),
        'high': np.array([r[4] for r in raw]),
        'low': np.array([r[5] for r in raw]),
        'close': np.array([r[6] for r in raw]),
        'divident': np.array([r[9] for r in raw]),
        'split': np.array([r[10] for r in raw])
        }

# for i in range(len(raw)):
#    raw2['date'][i] = datetime.strptime(raw[i][2], '%Y-%m-%d')
#    raw2['open'][i] = raw[i][3]
#    raw2['high'][i] = raw[i][4]
#    raw2['low'][i] = raw[i][5]
#    raw2['close'][i] = raw[i][6]
#    raw3['split'][i] = raw[i][10]

# for r in raw:
#    raw2['date'].append(r[2])
#    raw2['open'].append(r[3])
#    raw2['high'].append(r[4])
#    raw2['low'].append(r[5])
#    raw2['close'].append(r[6])
#    raw2['split'].append(r[10])

for i in range(len(raw2['date'])):

    s = raw2['split'][i]
    if s != 1.0:
        raw2['open'][:i] *= s
        raw2['close'][:i] *= s
        raw2['low'][:i] *= s
        raw2['high'][:i] *= s

    div = raw2['divident'][i]
    if div != 0.0:
        baz = raw2['close'][i - 1]
        foo = raw2['close'][i]
        bar = raw2['close'][i + 1]

        div_adj = (raw2['close'][i] + div) / raw2['close'][i]

        raw2['open'][:i] *= div_adj
        raw2['close'][:i] *= div_adj
        raw2['low'][:i] *= div_adj
        raw2['high'][:i] *= div_adj


df = pd.DataFrame(raw2)

# pprint(df.head())

data = [go.Candlestick(x=df['date'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'])]

fig_signal = go.Figure(data=data)
fig_signal.show()
