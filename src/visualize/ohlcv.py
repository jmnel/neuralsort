import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
import shutil
from pathlib import Path
from time import perf_counter
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FormatStrFormatter
import mplfinance as mpf
import pandas as pd
import progressbar

import settings

t_start = perf_counter()

PREDICT_WINDOW = 10
PLOT_LEN = 100

OUTPUT_DIR = Path(__file__).absolute().parent / 'plots'


pred_db = sqlite3.connect(settings.DATA_DIRECTORY / 'topk.sqlite3')
qdl_db = sqlite3.connect(settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME)

days = tuple(zip(*pred_db.execute('SELECT DISTINCT date FROM predicted;').fetchall()))[0][:-1]

days_path = Path.cwd() / 'trading_days.json'
if days_path.is_file():
    with open(days_path, 'r') as json_file:
        trading_days = json.loads(json_file.read())
else:
    trading_days = qdl_db.execute('SELECT DISTINCT date FROM qdl_eod WHERE date <= ? ORDER BY DATE',
                                  (days[-1],)).fetchall()
    trading_days = tuple(d[0] for d in trading_days)
    with open(days_path, 'w') as json_file:
        json_file.write(json.dumps(trading_days))


# if OUTPUT_DIR.is_dir():
#    shutil.rmtree(OUTPUT_DIR)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

for day_idx, day in enumerate(days):

    day_dir = OUTPUT_DIR / day

    if not day_dir.exists():
        day_dir.mkdir()

    start_date = trading_days[-PLOT_LEN + day_idx]

    symbols_predicted = tuple(zip(*pred_db.execute('SELECT symbol FROM predicted WHERE date=?;',
                                                   (day,)).fetchall()))[0]
    symbols_true = tuple(zip(*pred_db.execute('SELECT symbol FROM true WHERE date=?;',
                                              (day,)).fetchall()))[0]
    true_set = set(symbols_true)
    predicted_set = set(symbols_predicted)

    symbols = true_set.union(predicted_set)

    symbols -= {'VIX', '$VIX', 'SP500', '$SPX'}

    print(f'Rendering charts for {day_idx+1} of {len(days)} : {day}')

    bar = progressbar.ProgressBar(maxval=len(symbols))

    bar.start()
    for symbol_idx, symbol in enumerate(symbols):

        bar.update(symbol_idx)
        d, o, h, l, c, v = tuple(zip(*
                                     qdl_db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod
WHERE symbol=? AND date <= ? AND date >= ?
ORDER BY date;''', (symbol, day, start_date)).fetchall()[-PLOT_LEN:]))

        d = tuple(datetime.strptime(di, '%Y-%m-%d').date() for di in d)
        ho = tuple(h[idx] / o[idx] for idx in range(len(o)))
        data = {'Open': o,
                'High': h,
                'Low': l,
                'Close': c,
                'Volume': v,
                'High over open': ho}
        df = pd.DataFrame(data, index=d)

        c_norm = tuple(c[idx] / c[0] for idx in range(len(c)))

        title = f'{symbol}'
        if symbol in true_set and symbol in predicted_set:
            title += f' FISH / predicted + true'
        elif symbol in true_set:
            title += f' true'
        elif symbol in predicted_set:
            title += f' predicted'

        if symbol in true_set and symbol in predicted_set:
            prefix = 'f-'
        elif symbol in true_set:
            prefix = 't-'
        elif symbol in predicted_set:
            prefix = 'p-'

        ax1 = plt.subplot(3, 1, 1)
        plt.plot(d, ho, linewidth=0.5)
        plt.title('High / Open', fontsize=8)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.grid(True)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(d, o, linewidth=0.5)
        plt.plot(d, h, linewidth=0.5)
        plt.plot(d, l, linewidth=0.5)
        plt.plot(d, c, linewidth=0.5)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.ylabel('Price')
        plt.legend(('Open', 'High', 'Low', 'Close'), fontsize='x-small')
        plt.title('OHLC', fontsize=8)
        plt.grid(True)

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        plt3 = plt.bar(d, v)
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Shares')
        plt.title('Volume', fontsize=8)
        ax3.xaxis.set_major_formatter(DateFormatter('%b-%-d'))
        ax3.set_yticklabels(['{:.0e}K'.format(y) for y in ax3.get_yticks() * 1e-3])
        plt.grid(True)

        plt.gcf().suptitle(title, fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, left=0.15)

        if symbol in true_set and symbol in predicted_set:
            prefix = 'f-'
        elif symbol in true_set:
            prefix = 't-'
        elif symbol in predicted_set:
            prefix = 'p-'
        plt.savefig(fname=day_dir / f'{prefix}ho_{symbol}_{day}.png',
                    dpi=300,
                    pad_inches=0.25,
                    figsize=(800, 400))
        plt.clf()

        bar.update(symbol_idx + 1)

    bar.finish()
    print()


qdl_db.close()
pred_db.close()

timer = perf_counter() - t_start
mins = timer // 60.0
secs = timer % 60.0

print('Took {:2}:{:02} to run.'.format(int(mins), int(secs)))
