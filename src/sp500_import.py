import datetime
import csv
import sqlite3
import io
import pytz
import requests
import logging
import settings

logger = logging.getLogger(__name__)


def import_sp500():
    logger.info('Importing SP500 index.')
    endpoint = 'https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC'
    endpoint += '?period1={}&period2={}&interval=1d&events=history'

    period_start = -1325635200
    period_now = datetime.datetime.utcnow()
    period_end = period_now.replace(day=period_now.day - 1,
                                    hour=20,
                                    minute=0,
                                    second=0)
    yesterday = period_end.date().strftime('%Y-%m-%d')
    local = pytz.timezone('US/Eastern')
    period_end = int(period_end.timestamp())

    endpoint = endpoint.format(period_start, period_end)

    db_path = settings.DATA_DIRECTORY / settings.DATABASE_NAME

    response = requests.get(endpoint)
    assert response.status_code == 200

    reader = csv.reader(io.StringIO(response.text), delimiter=',')
    next(reader)
    rows = list(reader)
    rows = list((r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), int(r[6]))
                for r in rows)

    logger.info('Saving to database.')
    with sqlite3.connect(db_path) as db:

        db.execute('DELETE FROM qdl_symbols WHERE symbol="SP500";')
        db.execute('''
    INSERT INTO qdl_symbols(
        symbol,
        qdl_code,
        name,
        exchange,
        last_trade)
        VALUES(?, ?, ?, ?, ?);
    ''', ('SP500',
          '',
          'S&P500 Market Index',
          'INDEX',
          yesterday))
        db.commit()

        rows2 = list()
        for idx, r in enumerate(rows):
            date = r[0]
            o = r[1]
            h = r[2]
            l = r[3]
            c = r[4]
            v = r[6]
            c_adj = r[5]
            adj_ratio = c_adj / c
            o_adj = o * adj_ratio
            h_adj = h * adj_ratio
            l_adj = l * adj_ratio
            v_adj = v
            rows2.append((
                'SP500',
                date,
                o, h, l, c, v,
                0.0, 1.0,
                o_adj, h_adj, l_adj, c_adj, v_adj))

        db.executemany('''
    INSERT INTO qdl_eod(
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume,
        dividend,
        split,
        adj_open,
        adj_high,
        adj_low,
        adj_close,
        adj_volume)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', rows2)
        db.commit()
