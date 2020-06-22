import csv
import sqlite3
from pprint import pprint
from pathlib import Path
import os
import requests
import io
import zipfile
import re
import logging

import quandl

import settings
from ticker_filter import TickerFilter

quandl.ApiConfig.api_key = settings.QUANDL_API_KEY

logger = logging.getLogger(__name__)


def prepare_database(db: sqlite3.Connection):

    logger.info(f'Preparing database {settings.DATABASE_NAME}...')

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('DROP TABLE IF EXISTS iex_trade_reports;')
    db.execute('DROP TABLE IF EXISTS iex_days;')
    db.execute('DROP TABLE IF EXISTS iex_symbols;')

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

    logger.info('Done')


def prep2(db: sqlite3.Connection):

    db.execute('''
CREATE TABLE IF NOT EXISTS iex_days_symbols(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    FOREIGN KEY(day) REFERENCES iex_days(date),
    FOREIGN KEY(symbol) REFERENCES iex_symbols(symbol)
);''')


def parse_csv(db: sqlite3.Connection, symbols_meta):
    """
    Parse all csv files for all available days.

    """

    logger.info('Parsing csv files for days.')

    csv_directory = settings.DATA_DIRECTORY / 'csv_test'

    # Get list of days by enumerating csv files in directory.
    for f in os.listdir(csv_directory):
        csv_path = csv_directory / f
        if csv_path.is_file and csv_path.suffix == '.csv':

            day = csv_path.name[:-4]
            day = '-'.join((day[:4], day[4:6], day[6:8]))

            db.execute('INSERT INTO iex_days(date) VALUES(?);', (day,))
            db.commit()
            day_id = db.execute('SELECT last_insert_rowid();').fetchone()[0]

            logger.info(f'Found day {day} @ {f}.')

            with open(csv_path, 'r') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                date_str = '-'.join((f[:4], f[4:6], f[6:8]))

                rows = list()
                for idx, row in enumerate(reader):
                    timestamp, symbol, price, size = row
                    qdl_symbol = symbol.replace('.', '_').replace('-', '_')
                    if qdl_symbol in symbols_meta:
                        rows.append((date_str, timestamp, qdl_symbol, price, size))

                logger.info(f'Storing {len(rows)} of {idx+1} messages to database.')

                db.executemany('''
INSERT INTO iex_trade_reports(day, timestamp, symbol, price, size)
VALUES(?, ?, ?, ?, ?);
''', rows)
                db.commit()


def get_quandl_meta(db):

    logger.info('Downloading Quandl EOD metadata.')

    META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'
    response = requests.get(META_ENDPOINT.format(settings.QUANDL_API_KEY))
    if not response.ok:
        logger.critical(f'Quandl metadata endpoint request error {response.status_code}.')
        exit()

    exchange_re = re.compile(r'<p><b>Exchange<\/b>: ([\w ]+)<\/p>')

    ss = io.BytesIO(response.content)
    zf = zipfile.ZipFile(ss)

    logger.info('Processing metadata')
    count = 0
    with zf.open('EOD_metadata.csv', 'r') as f:
        f2 = io.TextIOWrapper(f, encoding='utf-8')
        reader = csv.reader(f2, delimiter=',')
        next(reader)

        rows = list()
        for idx, row in enumerate(list(reader)):

            count += 1

            symbol = row[0]
            name = row[1][:-38 - len(row[0])]
            qdl_code = row[0]

            # Extract exchange from description column.
            m = exchange_re.search(row[2])
            if m:
                exchange = m.groups()[0]

            # Symbol AA_P has no description, exchange, or data.
            else:
                continue

            if TickerFilter.filter(symbol, name, exchange):
                rows.append((symbol, qdl_code, name, exchange))

    symbols_meta = {r[0]: r for r in rows}

    db.executemany('''
INSERT INTO iex_symbols(symbol, qdl_code, name, exchange)
VALUES(?, ?, ?, ?);
''', rows)
    db.commit()

    logger.info(f'Found {count} symbols, filtered to {len(rows)}')
    logger.info('Done')

    return symbols_meta


def main():
    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
        prep2(db)
        #        prepare_database(db)
        #        symbols_meta = get_quandl_meta(db)
        #        parse_csv(db, symbols_meta)


if __name__ == '__main__':
    main()
