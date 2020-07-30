import sqlite3
from pprint import pprint
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

import settings

IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

N = 20

db = sqlite3.connect(IEX_PATH)

days = tuple(zip(*db.execute('SELECT date FROM iex_days;').fetchall()))[0]
#days = days[:4]

indices = list()
for day in days:

    print(f'Enumerating symbols for day {day}')

    rows = db.execute('''
SELECT count(id), symbol FROM iex_trade_reports WHERE day=? GROUP BY SYMBOL;''', (day,)).fetchall()

    for r in rows:
        if r[0] > 2000:
            indices.append((day, r[1]))
            print(f'\t{r[1]} -> {r[0]}')


db.execute(f'''
CREATE TABLE IF NOT EXISTS iex_tick_bar_{N}(
    id PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    timestamp UNSIGNED BIG INT NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    FOREIGN KEY(day) REFERENCES iex_days(date),
    FOREIGN KEY(symbol) REFERENCES iex_symbols(symbol));''')

db.execute(f'CREATE INDEX IF NOT EXISTS iex_tick_bar_{N}_day_idx ON iex_tick_bar_{N}(day);')
db.execute(f'CREATE INDEX IF NOT EXISTS iex_tick_bar_{N}_symbol_idx ON iex_tick_bar_{N}(symbol);')
db.execute(f'CREATE INDEX IF NOT EXISTS iex_tick_bar_{N}_day_symbol_idx ON iex_tick_bar_{N}(day, symbol);')


for day, symbol in indices:

    rows = db.execute('''
SELECT timestamp, price, size FROM iex_trade_reports
WHERE day=? AND symbol=?
ORDER BY timestamp;''', (day, symbol)).fetchall()

    bars = list()

    t = 0
    o, h, l, c, v = 0, 0, 0, 0, 0
    for idx, (ts, price, size) in enumerate(rows):
        if idx % N == 0 and idx != 0:
            bars.append((t, o, h, l, c, v))
        if idx % N == 0:
            t = datetime.fromtimestamp(ts * 1e-9)
            o = price
            h = float('-inf')
            l = float('inf')
            c = 0
            v = 0
        if idx % N == N - 1:
            c = price

        h = max(price, h)
        l = min(price, l)
        v += size

    bars = tuple((day, symbol, b[0], b[1], b[2], b[3], b[4], b[5]) for b in bars)
    db.executemany(f'''
INSERT INTO iex_tick_bar_{N}(day, symbol, timestamp, open, high, low, close, volume)
VALUES(?, ?, ?, ?, ?, ?, ?, ?);''', bars)

    db.commit()

#    d, o, h, l, c, v = tuple(zip(*bars))
#    data = {'Open': o,
#            'High': h,
#            'Low': l,
#            'Close': c,
#            'Volume': v}
#    df = pd.DataFrame(data, index=d)

#    mpf.plot(df, type='candle', volume=True, style='yahoo')


db.close()
