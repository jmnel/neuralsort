import sqlite3
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import shutil
from pathlib import Path
from time import perf_counter
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import pandas_market_calendars as mcal

import settings

PREDICT_WINDOW = 10
PLOT_LEN = 100
OUTPUT_DIR = Path(__file__).absolute().parent / 'plots'


def connect_databases() -> Tuple[sqlite3.Connection, sqlite3.Connection]:
    prediction_db = sqlite3.connect(settings.DATA_DIRECTORY / 'topk.sqlite')
    qdl_db = sqlite3.connect(settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME)
    return prediction_db, qdl_db


def previous_n_trading_days(end_date: str, num_days: int):

    start_date = datetime.strptime(end_date, '%Y-%m-%d').date() - relativedelta(years=2)
    start_date = start_date.strftime('%Y-%m-%d')

    nyse = mcal.get_calendar('NYSE').schedule(start_date, end_date).iloc[-num_days:]
    nasdaq = mcal.get_calendar('NASDAQ').schedule(start_date, end_date).iloc[-num_days:]

    assert len(nyse) == num_days
    assert len(nasdaq) == num_days

    pd.testing.assert_frame_equal(nyse, nasdaq)

    for day in nyse.index.values:
        print(day.strftime('%Y-%m-%d'))


def main():
    previous_n_trading_days('2020-07-20', 5)


if __name__ == '__main__':
    main()
