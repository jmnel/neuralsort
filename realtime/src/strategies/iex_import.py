import csv
import sqlite3
from pprint import pprint
from pathlib import Path
import os
import requests
import io
import zipfile
import re

import quandl

import settings

quandl.ApiConfig.api_key = settings.QUANDL_API_KEY


def prepare_database() -> sqlite3.Connection:

    db = sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME)

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('DROP TABLE IF EXISTS iex_trade_reports;')
    db.execute('DROP TABLE IF EXISTS iex_days;')
    db.execute('DROP TABLE IF EXISTS iex_symbols;')
#    db.execute('DROP TABLE IF EXISTS iex_trade_reports_day_symbols;')

    # Create days meta table.
    db.execute('''
CREATE TABLE iex_days(
    date DATE PRIMARY KEY,
    message_count UNSIGNED_INTEGER DEFAULT 0);
''')

    # Create symbols meta table.
    db.execute('''
CREATE TABLE iex_symbols(
   symbol CHAR(16) PRIMARY KEY,
   qdl_code CHAR(16),
   name CHAR(256),
   exchange CHAR(16),
   is_common BOOL,
   last_trade DATE);
''')

    # Create trade reports table.
    db.execute('''
CREATE TABLE iex_trade_reports(
   id INTEGER PRIMARY KEY,
   day MEDIUMINT UNSIGNED NOT NULL,
   timestamp UNSIGNED BIG INT NOT NULL,
   symbol CHAR(16) NOT NULL,
   price INTEGER NOT NULL,
   size INTEGER NOT NULL,
   FOREIGN KEY(day) REFERENCES iex_days(date),
   FOREIGN KEY(symbol) REFERENCES iex_symbols(symbol)
);
''')

    return db


def parse_csv(db):
    csv_directory = settings.DATA_DIRECTORY / 'csv_test'

    # Get list of days be enumerating CSV files in directory.
#    days = list()
#    for f in os.listdir(csv_directory):
#        csv_path = csv_directory / f
#        if csv_path.is_file and csv_path.suffix == '.csv':
#            days.append(csv_path.name)

#    pprint(days)
#    exit()

    for f in os.listdir(csv_directory):
        csv_path = csv_directory / f
        if csv_path.is_file and csv_path.suffix == '.csv':

            day = csv_path.name[:-4]
            day = '-'.join((day[:4], day[4:6], day[6:8]))

            print(day)

            db.execute('INSERT INTO iex_days(date) VALUES(?);', (day,))
            db.commit()
            day_id = db.execute('SELECT last_insert_rowid();').fetchone()[0]
            print(day_id)

            print('\tEnumerating symbols...')
            with open(csv_path, 'r') as csv_file:
                rows = csv.reader(csv_file, delimiter=',')
                _, rows, _, _ = zip(*list(rows))

            rows = set(rows)
            rows = tuple((r,) for r in rows)
            db.executemany('''
INSERT OR IGNORE INTO iex_symbols(symbol) VALUES(?)
;''', rows)
            db.commit()
            print('\tDone.')

            with open(csv_path, 'r') as csv_file:
                rows = csv.reader(csv_file, delimiter=',')
                date_str = '-'.join((f[:4], f[4:6], f[6:8]))
                rows = list(rows)

                rows = list((date_str, r[0], r[1], r[2], r[3]) for r in rows)

                db.executemany('''
INSERT INTO iex_trade_reports(day, timestamp, symbol, price, size)
VALUES(?, ?, ?, ?, ?);
''', rows)
                db.commit()


def get_quandl_ticker_info(db):

    META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'
    response = requests.get(META_ENDPOINT.format(settings.QUANDL_API_KEY))
    if not response.ok:
        print('error')

#    pprint(response.content)

    re.compile(r':

    ss=io.BytesIO(response.content)
    zf=zipfile.ZipFile(ss)
    with zf.open('EOD_metadata.csv', 'r') as f:
        f2=io.TextIOWrapper(f, encoding='utf-8')
        reader=csv.reader(f2, delimiter=',')
        h=next(reader)
        pprint(h)
        rows=tuple(reader)
        pprint(rows[:5])
        rows=tuple((r[0],
                      r[0],
                      r[1][:-38 - len(r[0])]) for r in rows)
        pprint(rows[:5])
#        pprint(rows[0][1])
        exit()

        #    response.ok()

        #    reader = csv.reader(io.StringIO(tickers_reponse.text), delimiter=',')

        # This the url of the API endpoint for the master list of tickers.
        #    TICKER_URL = 'https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv'

        # Get the csv file using HTTP request.
        #    tickers_reponse = requests.get(TICKER_URL)

        #    reader = csv.reader(io.StringIO(tickers_reponse.text), delimiter=',')
        #    h = next(reader)
        #    pprint(h)
        #    ticker_list = list(reader)

        #    ticker_dict = {r[0]: (r[1], r[2], r[3], r[4]) for r in ticker_list}

        #    pprint(ticker_dict)

        #    symbols = tuple(zip(*db.execute('SELECT symbol from iex_symbols;').fetchall()))[0]

        #    for s in symbols:
        #        if s not in ticker_dict:
        #            print(f'missing: {s}')

        # Filter out non-common stocks.
        #    ticker_list = list(
        #        filter(lambda row: TickerFilter.filter(row[0], row[2], row[3]), ticker_list))


def main():
    db=prepare_database()
#    parse_csv(db)
    get_quandl_ticker_info(db)
    db.close()


if __name__ == '__main__':
    main()
