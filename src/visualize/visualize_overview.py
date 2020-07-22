import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd

import settings

t_start = perf_counter()

PREDICT_WINDOW = 10
PLOT_LEN = 100

OUTPUT_DIR = Path(__file__).absolute().parent / 'plots'


pred_db = sqlite3.connect(settings.DATA_DIRECTORY / 'topk.sqlite3')
qdl_db = sqlite3.connect(settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME)

days = tuple(zip(*pred_db.execute('SELECT DISTINCT date FROM predicted;').fetchall()))[0][:-1]

trading_days = qdl_db.execute('SELECT DISTINCT date FROM qdl_eod WHERE date <= ? ORDER BY DATE',
                              (days[-1],)).fetchall()
trading_days = tuple(d[0] for d in trading_days)

for idx in range(len(days)):
    print(f'a: {trading_days[-len(days)+idx]}')
    print(f'b: {days[idx]}')
assert trading_days[-len(days) + idx] == days[idx]

# if OUTPUT_DIR.is_dir():
#    shutil.rmtree(OUTPUT_DIR)

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

for day_idx, day in enumerate(days[:1]):

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

#    plt.show()

    for symbol in symbols:

        print(f'Getting EOD values for {symbol}.')
        d, o, h, c, v = tuple(zip(*
                                  qdl_db.execute('''
SELECT date, adj_open, adj_high, adj_close, adj_volume
FROM qdl_eod
WHERE symbol=? AND date <= ? AND date >= ?
ORDER BY date;''', (symbol, day, start_date)).fetchall()[-PLOT_LEN:]))

        d = tuple(datetime.strptime(di, '%Y-%m-%d').date() for di in d)
        ho = tuple(h[idx] / o[idx] for idx in range(len(o)))
        data = {'Open': o,
                'High': h,
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

#        ax = gca()
#        ax.xaxis.set_
#        plt.plot(d, c_norm, lw=0.5)
#        plt.scatter(d, c_norm, s=0.5)
#        plt.format_xdata = mdates.DateFormatter('%b-%d')
        plt.scatter(tuple(range(0, len(ho))), ho, s=0.5)
        plt.xticks(rotation=45)
        plt.title(title)

    plt.show()

#        mpf.plot(df, type='candle', mav=[4, 12, 26], volume=True, style='yahoo',
#                 tight_layout=True,
#                 title=title,
#                 savefig={'fname': day_dir / f'{prefix}ohlcv_{symbol}_{day}.png',
#                          'dpi': 200,
#                          'pad_inches': 0.25,
#                          'figsize': (600, 400)}
#                 )

#        plt.plot(data['Open'])
#        plt.show()


qdl_db.close()
pred_db.close()

timer = perf_counter() - t_start
mins = timer // 60.0
secs = timer % 60.0

print('Took {:2}:{:02} to run.'.format(int(mins), int(secs)))
