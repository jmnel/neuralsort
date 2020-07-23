import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
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

if OUTPUT_DIR.is_dir():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir()

for day in days:

    day_dir = OUTPUT_DIR / day
    day_dir.mkdir()

    symbols_predicted = tuple(zip(*pred_db.execute('SELECT symbol FROM predicted WHERE date=?;',
                                                   (day,)).fetchall()))[0]
    symbols_true = tuple(zip(*pred_db.execute('SELECT symbol FROM true WHERE date=?;',
                                              (day,)).fetchall()))[0]
    true_set = set(symbols_true)
    predicted_set = set(symbols_predicted)

    symbols = true_set.union(predicted_set)

    symbols -= {'VIX', '$VIX', 'SP500', '$SPX'}

    for symbol in symbols:

        print(f'Getting EOD values for {symbol}.')
        d, o, h, l, c, v = tuple(zip(*
                                     qdl_db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod
WHERE symbol=? AND date <= ?
ORDER BY date;''', (symbol, day)).fetchall()[-PLOT_LEN:]))

        d = tuple(datetime.strptime(di, '%Y-%m-%d') for di in d)
        data = {'Open': o,
                'High': h,
                'Low': l,
                'Close': c,
                'Volume': v}
        df = pd.DataFrame(data, index=d)

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

        mpf.plot(df, type='candle', mav=[4, 12, 26], volume=True, style='yahoo',
                 tight_layout=True,
                 title=title,
                 savefig={'fname': day_dir / f'{prefix}ohlcv_{symbol}_{day}.png',
                          'dpi': 200,
                          'pad_inches': 0.25,
                          'figsize': (600, 400)}
                 )

#        plt.plot(data['Open'])
#        plt.show()


qdl_db.close()
pred_db.close()

timer = perf_counter() - t_start
mins = timer // 60.0
secs = timer % 60.0

print('Took {:2}:{:02} to run.'.format(int(mins), int(secs)))
