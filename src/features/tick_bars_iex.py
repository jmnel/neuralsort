import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
from pathlib import Path
import shutil

import pandas as pd
import numpy as np

import settings

NB_TICKS = 20

IEX_DB_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME
RL_DB_PATH = settings.DATA_DIRECTORY / (settings.RL_DATABSE_NAME + '-2')

iex_db = sqlite3.connect(IEX_DB_PATH)


if RL_DB_PATH.is_file():
    RL_DB_PATH.unlink()
rl_db = sqlite3.connect(RL_DB_PATH)


rl_db.execute(f'DROP INDEX IF EXISTS ticks_{NB_TICKS}_day_idx;')
rl_db.execute(f'DROP INDEX IF EXISTS ticks_{NB_TICKS}_symbol_idx;')
rl_db.execute(f'DROP TABLE IF EXISTS ticks_{NB_TICKS}_meta;')
rl_db.execute(f'DROP TABLE IF EXISTS ticks_{NB_TICKS};')
rl_db.execute(f'''
CREATE TABLE ticks_{NB_TICKS}_meta(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    count UNSIGNED INTEGER);''')
rl_db.execute(f'''
CREATE TABLE ticks_{NB_TICKS}(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    idx INTEGER NOT NULL,
    timestamp FLOAT NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL);''')

rl_db.execute(f'CREATE INDEX ticks_{NB_TICKS}_day_idx ON ticks_{NB_TICKS}(day);')
rl_db.execute(f'CREATE INDEX ticks_{NB_TICKS}_symbol_idx ON ticks_{NB_TICKS}(symbol);')
rl_db.close()

rows = iex_db.execute('''
SELECT day, symbol, message_count FROM iex_ticks_meta WHERE message_count>=2000;
''').fetchall()

#rows = rows[-1800:-1500]
#rows = rows[-2000:]

for idx, (day, symbol, message_count) in enumerate(rows):

    print(f'{idx+1} of {len(rows)}')

    rows2 = iex_db.execute('''
SELECT timestamp, price, size FROM iex_trade_reports
WHERE day=? AND symbol=?;''',
                           (day, symbol)).fetchall()

#    if symbol == 'NKLA' and day == '2020-07-06':
#        print(f'{idx+1} of {len(rows)}')

    if len(rows2) == 0:
        print(f'Error no rows: {symbol}')
        continue

    data = list()
    timestamp, price, size = tuple(zip(*rows2))
    timestamp = tuple(t * 1e-9 for t in timestamp)

    for idx in range(0, len(rows2), NB_TICKS):
        t0, t1 = idx, idx + NB_TICKS
        o = price[idx]
        h = np.max(price[idx:idx + NB_TICKS])
        assert np.isfinite(h)
        assert len(price[idx:idx + NB_TICKS]) > 0
        l = np.min(price[idx:idx + NB_TICKS])
        c = price[idx:idx + NB_TICKS][-1]
        v = int(np.sum(size[idx:idx + NB_TICKS]) * 100)
        t = timestamp[idx:idx + NB_TICKS][-1]

        if symbol == 'NKLA' and day == '2020-07-06':
            print('{} -> {:.3f} - {:.3f} - {:.3f} - {:.3f} - {}'.format(idx, o, h, l, c, v))
#            done = True

        data.append((day, symbol, idx / NB_TICKS, t, o, h, l, c, v))

    rl_db = sqlite3.connect(RL_DB_PATH)
    rl_db.execute(f'''
INSERT INTO ticks_{NB_TICKS}_meta(day, symbol, count)
VALUES(?, ?, ?)''', (day, symbol, len(data)))

    rl_db.executemany(f'''
INSERT INTO ticks_{NB_TICKS}(
    day, symbol, idx, timestamp, open, high, low, close, volume)
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)

    rl_db.commit()
    rl_db.close()


iex_db.close()
