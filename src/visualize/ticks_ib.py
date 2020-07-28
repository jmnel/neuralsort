import sqlite3
from pprint import pprint
from datetime import datetime, time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter
import mplfinance as mpf
import pandas as pd

import settings

OUTPUT_DIR = settings.DATA_DIRECTORY / 'tick_plots'

IB_DB_PATH = settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME

db = sqlite3.connect(IB_DB_PATH)

days = tuple(zip(*db.execute('SELECT date FROM ib_days;').fetchall()))[0]
days = days[1:]

if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir()

for idx, day in enumerate(days):
    symbols = tuple(zip(*db.execute('''
SELECT distinct(symbol) FROM ib_trade_reports WHERE day=?;
''', (day,)).fetchall()))[0]

    day_dir = OUTPUT_DIR / day
    if not day_dir.is_dir():
        day_dir.mkdir()

    for jdx, symbol in enumerate(symbols):

        ts, p, s = tuple(zip(*db.execute('''
SELECT timestamp, price, size FROM ib_trade_reports
WHERE day=? AND symbol=?
ORDER BY timestamp;''', (day, symbol)).fetchall()))

        t = tuple(datetime.fromtimestamp(x * 1e-9) for x in ts)
        p = tuple(x * 1e-2 for x in p)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(t, p, linewidth=0.4)
        ax.set_xlabel('Time', fontsize='small')
        ax.set_ylabel('Price', fontsize='small')
        plt.suptitle(f'{symbol}')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(DateFormatter('%-I:%M'))
        plt.savefig(fname=day_dir / f'{symbol}_ib_tick_{day}.png',
                    dpi=300,
                    pad_inches=0.25,
                    figsize=(800, 400))
        plt.clf()
        plt.close()


db.close()
