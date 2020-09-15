import sqlite3
from pprint import pprint
from datetime import datetime, timedelta, time
from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import mplfinance as mpf
plt.style.use('ggplot')
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np

import settings

IB_DB_PATH = settings.DATA_DIRECTORY / (settings.IB_DATABASE_NAME)

end_day = datetime.now().date()
end_day -= timedelta(days=1)

end_day = end_day.strftime('%Y-%m-%d')

print(end_day)

db = sqlite3.connect(IB_DB_PATH)

days = tuple(zip(*db.execute('SELECT date FROM ib_days WHERE date==?;',
                             (end_day,)).fetchall()))[0]

for day in days:

    day_dir = Path(__file__).absolute().parent / 'plots' / day
    if Path.is_dir(day_dir):
        shutil.rmtree(day_dir)
    day_dir.mkdir()

    symbols = db.execute('''
SELECT DISTINCT(symbol) FROM ib_trade_reports WHERE day=?;''',
                         (day,)).fetchall()
    symbols = tuple(r[0] for r in symbols)

    day_p = datetime.strptime(day, '%Y-%m-%d')
    start = day_p + timedelta(hours=9, minutes=25, seconds=0)
    end = day_p + timedelta(hours=12 + 4, minutes=5, seconds=0)

    for symbol in symbols:

        rows = db.execute('''
SELECT timestamp, price FROM ib_trade_reports
WHERE day=? AND symbol=? AND timestamp != 0 ORDER BY timestamp;''', (day, symbol)).fetchall()

        ts, price = tuple(zip(*rows))
        ts = tuple(datetime.fromtimestamp(t) for t in ts)
        price = tuple(p * 1e-2 for p in price)

        xy = tuple(zip(ts, price))
        xy = tuple(filter(lambda x: x[0].time() >= time(9, 30, 0) and x[0].time() <= time(16, 0, 0), xy))
        ts, price = tuple(zip(*xy))

        fig, ax = plt.subplots(1, 1)
        ax.plot(ts, price, linewidth=0.4)
        ax.set_xlabel('Time', fontsize='medium')
        ax.set_ylabel('Price', fontsize='medium')
        plt.suptitle(f'{symbol}')
        plt.xticks(rotation=45)
        ax.xaxis.set_major_formatter(DateFormatter('%-I:%M'))
        plt.xlim((start, end))
        plt.savefig(fname=day_dir / f'{symbol}_ib_tick_{day}.png',
                    dpi=300,
                    pad_inches=0.25,
                    figsize=(800, 400))
        plt.close()

        rows2 = db.execute('''
SELECT timestamp, open, high, low, close, volume
FROM ticks_5 WHERE symbol=? AND day=?
ORDER BY idx;''', (symbol, day)).fetchall()

        if len(rows2) == 0:
            continue
        rows2 = tuple((datetime.fromtimestamp(r[0]), *r[1:]) for r in rows2)
        rows2 = tuple(filter(lambda x: x[0].time() >= time(9, 30, 0) and x[0].time() <= time(16, 0, 0), rows2))

        ts, o, h, l, c, v = tuple(zip(*rows2))
        if len(ts) == 0:
            continue

#        ts = tuple(datetime.fromtimestamp(t) for t in ts)

        data = {'Open': o,
                'High': h,
                'Low': l,
                'Close': c,
                'Volume': v}
        df = pd.DataFrame(data, index=ts)

        mpf.plot(df,
                 type='candle',
                 mav=[4, 12, 26],
                 volume=True,
                 tight_layout=True,
                 style='yahoo',
                 title=f'{symbol} : 5-tick bars',
                 savefig={'fname': day_dir / f'{symbol}_ib_ohlcv_{day}.png',
                          'dpi': 200,
                          'pad_inches': 0.25,
                          'figsize': (600, 400)})
        plt.close()

db.close()
