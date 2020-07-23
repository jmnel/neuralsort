import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import shutil
from pathlib import Path
from time import perf_counter
from typing import Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import itertools
from multiprocessing import Pool
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplfinance as mpf
import pandas as pd
import pandas_market_calendars as mcal
#import progressbar
import tqdm

import settings

PREDICT_WINDOW = 10
PLOT_LEN = 100
OUTPUT_DIR = Path(__file__).absolute().parent / 'plots'
NUM_WORKERS = 8


def previous_n_trading_days(end_date: str, num_days: int):

    start_date = datetime.strptime(end_date, '%Y-%m-%d').date() - relativedelta(years=2)
    start_date = start_date.strftime('%Y-%m-%d')

    nyse = mcal.get_calendar('NYSE').schedule(start_date, end_date).iloc[-num_days:]
    nasdaq = mcal.get_calendar('NASDAQ').schedule(start_date, end_date).iloc[-num_days:]

    assert len(nyse) == num_days
    assert len(nasdaq) == num_days

    pd.testing.assert_frame_equal(nyse, nasdaq)

    return tuple(pd.to_datetime(day).strftime('%Y-%m-%d') for day in nyse.index.values)


#num_tasks = 0
#task_idx = itertools.count()
#bar = None


def run_tasks():

    with sqlite3.connect(settings.DATA_DIRECTORY / 'topk.sqlite3') as db:
        prediction_days = tuple(zip(*db.execute('SELECT DISTINCT date FROM predicted;')
                                    .fetchall()))[0][:-1]
    tasks = list()
    for day in prediction_days:

        with sqlite3.connect(settings.DATA_DIRECTORY / 'topk.sqlite3') as db:
            true_symbols = tuple(zip(*db.execute('''
SELECT DISTINCT symbol FROM true WHERE date=?;
''', (day,)).fetchall()))[0][:-1]
            predicted_symbols = tuple(zip(*db.execute('''
SELECT DISTINCT symbol FROM predicted WHERE date=?;
''', (day,)).fetchall()))[0][:-1]

        true_symbols = set(true_symbols) - {'VIX', 'SP500', '$VIX', '$SPX'}
        predicted_symbols = set(predicted_symbols) - {'VIX', 'SP500', '$VIX', '$SPX'}
        symbols = true_symbols.union(predicted_symbols)

        day_dir = OUTPUT_DIR / day
        if day_dir.exists():
            shutil.rmtree(day_dir)
        day_dir.mkdir()

        for symbol in symbols:
            tasks.append((day, symbol, symbol in true_symbols, symbol in predicted_symbols))

#    num_tasks = len(tasks)

#    global bar
#    bar = progressbar.ProgressBar(maxval=num_tasks)
#    bar.start()
#    futures = list()
    with Pool(NUM_WORKERS) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(render_ohlcv,
                                               tasks), total=len(tasks)):
            pass
#        pool.map(render_ohlcv, task)


#    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
#        for task in tasks:
#            futures.append(pool.submit(render_ohlcv, *task))

#    for task in futures:
#        task.result()

#    bar.finish()


def render_ohlcv(task):
    day, symbol, is_true, is_predicted = task
    trading_days = previous_n_trading_days(day, PLOT_LEN)

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME) as db:
        d, o, h, l, c, v = tuple(zip(*db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod
WHERE symbol=? AND date >= ? AND date <= ?
ORDER BY date;''', (symbol, trading_days[0], day)).fetchall()))

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
    if is_true and is_predicted:
        title += f' - predicted & true'
        prefix = 'f-'
    elif is_true:
        title += f' - true'
        prefix = 't-'
    elif is_predicted:
        title += f' - predicted'
        prefix = 'p-'

    plt.ioff()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(d, ho, linewidth=0.5)
    ax1.set_title('High / Open', fontsize=8)
    ax1.grid(True)

    ax2.plot(d, o, linewidth=0.5)
    ax2.plot(d, h, linewidth=0.5)
    ax2.plot(d, l, linewidth=0.5)
    ax2.plot(d, c, linewidth=0.5)
    ax2.set_ylabel('Price')
    ax2.legend(('Open', 'High', 'Low', 'Close'), fontsize='x-small')
    ax2.set_title('OHLC', fontsize=8)
    ax2.grid(True)

    plt3 = ax3.bar(d, v)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Shares')
    ax3.set_title('Volume', fontsize=8)
    ax3.xaxis.set_major_formatter(DateFormatter('%b-%-d'))
    ax3.set_yticklabels(['{:.0e}K'.format(y) for y in ax3.get_yticks() * 1e-3])
    ax3.grid(True)

    fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.15)

    day_dir = OUTPUT_DIR / day

    fig.savefig(fname=day_dir / f'{prefix}{symbol}_ho_{day}.png',
                dpi=300,
                pad_inches=0.25,
                figsize=(800, 400))

    plt.close()


def main():

    start_time = perf_counter()
    run_tasks()

    timer = perf_counter() - start_time
    print('Took {:2}:{:02} to run.'.format(int(timer // 60.0), int(timer % 60.0)))


if __name__ == '__main__':
    main()
