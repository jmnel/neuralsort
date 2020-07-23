import sqlite3
from pprint import pprint
from datetime import datetime, timedelta, timezone
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
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplfinance as mpf
import pandas as pd
import pandas_market_calendars as mcal
import tqdm
import seaborn as sb
import numpy as np
import scipy

import settings

PREDICT_WINDOW = 10
PLOT_LEN = 100
OUTPUT_DIR = Path(__file__).absolute().parent / 'plots'
NUM_WORKERS = 8

IEX_DB_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME


def previous_n_trading_days(end_date: str, num_days: int):

    start_date = datetime.strptime(end_date, '%Y-%m-%d').date() - relativedelta(years=2)
    start_date = start_date.strftime('%Y-%m-%d')

    nyse = mcal.get_calendar('NYSE').schedule(start_date, end_date).iloc[-num_days:]
    nasdaq = mcal.get_calendar('NASDAQ').schedule(start_date, end_date).iloc[-num_days:]

    assert len(nyse) == num_days
    assert len(nasdaq) == num_days

    pd.testing.assert_frame_equal(nyse, nasdaq)

    return tuple(pd.to_datetime(day).strftime('%Y-%m-%d') for day in nyse.index.values)


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

        plot_num_ticks(day, symbols, true_symbols, predicted_symbols)

        for symbol in symbols:
            tasks.append((day, symbol, symbol in true_symbols, symbol in predicted_symbols))

    with Pool(NUM_WORKERS) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(render_ohlcv,
                                               tasks[:10]), total=len(tasks)):
            pass


def render_ohlcv(task):

    day, symbol, is_true, is_predicted = task

    with sqlite3.connect(IEX_DB_PATH) as db:
        rows = db.execute('''
SELECT timestamp, price, size FROM iex_trade_reports
WHERE day=? AND symbol=?
ORDER BY timestamp;''', (day, symbol)).fetchall()

    if len(rows) < 1:
        print(f'{symbol} has no trades on {day}.')
        sys.stdout.flush()
        return

    timestamp, price, size = zip(*rows)

#    seconds = tuple(t * 1e-9 for t in timestamp)

    date = tuple(datetime.utcfromtimestamp(t * 1e-9) - timedelta(hours=5) for t in timestamp)

    title = f'{symbol} / Ticks / '
    if is_true and is_predicted:
        title += f'predicted & true'
        prefix = 'f-'
    elif is_true:
        title += f'true'
        prefix = 't-'
    elif is_predicted:
        title += f'predicted'
        prefix = 'p-'

#    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(date, price, linewidth=0.5)

    xlimit = (min(date) - timedelta(minutes=10), max(date) + timedelta(minutes=10))

    ax1.set_xlim(xlimit)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax1.set_ylabel('Price')

    ax1.grid(True)

    plt2 = ax2.scatter(date, size, s=3)
    ax2.set_ylabel('Size (num. shares)')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.15)

    day_dir = OUTPUT_DIR / day

    fig.savefig(fname=day_dir / f'{prefix}{symbol}_ticks_{day}.png',
                dpi=300,
                pad_inches=0.25,
                figsize=(800, 400))

    plt.close()


def plot_num_ticks(day, symbols, true_symbols, predicted_symbols):

    print(f'Plotting tick counts for {day}')

    counts = list()
    counts_t = list()
    counts_p = list()

    with sqlite3.connect(IEX_DB_PATH) as db:

        for symbol in symbols:
            n = db.execute('''
SELECT COUNT(id) FROM iex_trade_reports
WHERE day=? AND symbol=?;''',
                           (day, symbol)).fetchall()[0][0]

            counts.append((symbol, n))
            if symbol in true_symbols:
                counts_t.append((symbol, n))
            if symbol in predicted_symbols:
                counts_p.append((symbol, n))

    n1 = tuple(n[1] for n in counts)
    n2 = tuple(n[1] for n in counts_t)
    n3 = tuple(n[1] for n in counts_p)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    sb.distplot(n1, bins=100, ax=ax1, norm_hist=False, kde=False, color='C1')
    ax1.set_title('All symbols', fontsize='small')
    ax1.set_ylabel('Frequency', fontsize='x-small')

    sb.distplot(n2, bins=100, ax=ax2, norm_hist=False, kde=False, color='C2')
    ax2.set_title('True top-k', fontsize='small')
    ax2.set_ylabel('Frequency', fontsize='x-small')

    sb.distplot(n3, bins=100, ax=ax3, norm_hist=False, kde=False, color='C3')
    ax3.set_title('Predicted top-k', fontsize='small')
    fig.suptitle(f'Tick count: {day}')
    ax3.set_xlabel('Num. ticks', fontsize='x-small')
    ax3.set_ylabel('Frequency', fontsize='x-small')

    fig.subplots_adjust(hspace=0.4)

    day_dir = OUTPUT_DIR / day
    fig.savefig(fname=day_dir / f'tick_count_{day}.png',
                dpi=300,
                pad_inches=0.25,
                figsize=(800, 400))

    print('Done')


def main():

    start_time = perf_counter()
    run_tasks()

    timer = perf_counter() - start_time
    print('Took {:2}:{:02} to run.'.format(int(timer // 60.0), int(timer % 60.0)))


if __name__ == '__main__':
    main()
