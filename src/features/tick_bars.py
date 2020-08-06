import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
mpl.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

import settings

NB_TICKS = 20

IB_DB_PATH = settings.DATA_DIRECTORY / (settings.IB_DATABASE_NAME + '-2')

end_day = datetime.now().date()
end_day -= timedelta(days=1)

end_day = end_day.strftime('%Y-%m-%d')

db = sqlite3.connect(IB_DB_PATH)

days = tuple(zip(*db.execute('SELECT date FROM ib_days WHERE date<=?;',
                             (end_day,)).fetchall()))[0]


db.execute(f'DROP INDEX IF EXISTS ticks_{NB_TICKS}_day_idx;')
db.execute(f'DROP INDEX IF EXISTS ticks_{NB_TICKS}_symbol_idx;')
db.execute(f'DROP TABLE IF EXISTS ticks_{NB_TICKS}_meta;')
db.execute(f'DROP TABLE IF EXISTS ticks_{NB_TICKS};')
db.execute(f'''
CREATE TABLE ticks_{NB_TICKS}_meta(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    count UNSIGNED INTEGER);''')
db.execute(f'''
CREATE TABLE ticks_{NB_TICKS}(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    idx INTEGER NOT NULL,
    timestamp UNSIGNED BIG INT NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume INTEGER NOT NULL);''')

db.execute(f'CREATE INDEX ticks_{NB_TICKS}_day_idx ON ticks_{NB_TICKS}(day);')
db.execute(f'CREATE INDEX ticks_{NB_TICKS}_symbol_idx ON ticks_{NB_TICKS}(symbol);')

for day in days:
    symbols = tuple(zip(*db.execute('''
SELECT DISTINCT(symbol) from ib_trade_reports WHERE day=? ORDER BY symbol;
''', (day,)).fetchall()))
    if len(symbols) == 0:
        continue

    symbols = symbols[0]

    for symbol in symbols:
        rows = db.execute('''
SELECT timestamp, price, size FROM ib_trade_reports
WHERE day=? AND symbol=? AND timestamp!=0 ORDER BY id;''', (day, symbol)).fetchall()

        if len(rows) == 0:
            print(symbol)
            continue

        if rows[0][0] > 1596636729 * 100:
            rows = list((int(((r[0] * 1e-9) + 0.5)), *r[1:]) for r in rows)

#        if len(rows) < 1000:
#            continue

        data = list()
        timestamp, price, size = tuple(zip(*rows))
        price = tuple(np.round(p * 1e-2, 2) for p in price)
        for idx in range(0, len(rows), NB_TICKS):
            o = price[idx]
            h = np.max(price[idx:idx + NB_TICKS])
            l = np.min(price[idx:idx + NB_TICKS])
            c = price[idx:idx + NB_TICKS][-1]
            v = int(np.sum(size[idx:idx + NB_TICKS]) * 100)
            t = timestamp[idx:idx + NB_TICKS][-1]
            data.append((day, symbol, idx / NB_TICKS, t, o, h, l, c, v))

        db.execute(f'''
INSERT INTO ticks_{NB_TICKS}_meta(day, symbol, count)
VALUES(?, ?, ?)''', (day, symbol, len(data)))

        db.executemany(f'''
INSERT INTO ticks_{NB_TICKS}(
    day, symbol, idx, timestamp, open, high, low, close, volume)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)

        db.commit()


db.close()
