import io
import os
from time import perf_counter
import requests
import csv
import sqlite3
import re
from datetime import datetime
import shutil
import zipfile
from tempfile import mkdtemp
from pathlib import Path
from pprint import pprint
import numpy as np
import quandl
#import matplotlib
# matplotlib.use('Qt5Cairo')
#import matplotlib.pyplot as plt

from ticker_filter import TickerFilter
import settings


def prepare_database(db: sqlite3.Connection):

    db.execute('PRAGMA foreign_keys = ON;')

    db.execute('''
DROP INDEX IF EXISTS qdl_eod_symbols_index;
''')

    db.execute('''
DROP TABLE IF EXISTS qdl_eod;
''')

    db.execute('''
DROP TABLE IF EXISTS qdl_symbols;
''')

    # Create meta table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_symbols(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL UNIQUE,
    qdl_code CHAR(32) NOT NULL,
    name CHAR(256) NOT NULL,
    exchange CHAR(16) NOT NULL,
    last_trade DATE NOT NULL,
    start_date DATE,
    end_date DATE,
    lifetime_returns FLOAT);
''')

    # Create data table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_eod(
    id INTEGER PRIMARY KEY,
    symbol_id MEDIUMINT UNSIGNED NOT NULL,
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    dividend FLOAT NOT NULL,
    split FLOAT NOT NULL,
    adj_open FLOAT NOT NULL,
    adj_high FLOAT NOT NULL,
    adj_low FLOAT NOT NULL,
    adj_close FLOAT NOT NULL,
    adj_volume FLOAT NOT NULL,
    FOREIGN KEY(symbol_id) REFERENCES qdl_symbols(id)
);''')

    # Create symbol view.
    db.execute('''
CREATE INDEX IF NOT EXISTS qdl_eod_symbols_index ON qdl_eod(symbol_id);
''')


def get_quandl_tickers(db: sqlite3.Connection):

    # This the url of the API endpoint for the master list of tickers.
    TICKER_URL = 'https://s3.amazonaws.com/quandl-production-static/end_of_day_us_stocks/ticker_list.csv'

    # Get the csv file using HTTP request.
    tickers_reponse = requests.get(TICKER_URL)

    reader = csv.reader(io.StringIO(tickers_reponse.text), delimiter=',')
    next(reader)
    ticker_list = list(reader)

    # Filter out non-common stocks.
    ticker_list = list(
        filter(lambda row: TickerFilter.filter(row[0], row[2], row[3]), ticker_list))

    # Insert list of symbols into meta table.
    db.executemany('''
INSERT INTO qdl_symbols(symbol, qdl_code, name, exchange, last_trade)
VALUES(?, ?, ?, ?, ?);
''', ticker_list)

    db.commit()


def bulk_download(db: sqlite3.Connection):

    REDOWNLOAD = False

    eod_path = settings.DATA_DIRECTORY / 'quandl' / 'EOD.zip'

    # Download file if it is missing or flag is set.
    if REDOWNLOAD or not eod_path.is_file():
        print('\tDownloading bulk EOD data file...')

        quandl.bulkdownload('EOD',
                            api_key=settings.QUANDL_API_KEY,
                            download_type='complete')

        shutil.move('EOD.zip', eod_path)

        print('\tDone')
    else:
        print('\talready downloaded')

    # Create temporary directory.
    print('\tExtracting zip file...')
    temp_dir = Path(mkdtemp(prefix='quandl_eod'))

    # Extract zip file.
    with zipfile.ZipFile(eod_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    print('\tParsing CSV file...')

    # Get path to csv file in temporary directory.
    csv_name = os.listdir(temp_dir)[0]
    csv_path = Path(temp_dir).absolute() / csv_name

    # Get list of symbols to keep.
    tickers = db.execute('''
SELECT symbol, id, qdl_code FROM qdl_symbols;
''').fetchall()

    tickers = {row[2].split('/')[1]: (row[1], row[0]) for row in tickers}

    # Process the bulk downloaded CSV file; we have to perform buffered write into database,
    # to limit memory consumption.
    with open(csv_path, newline='') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        prev_qdl_code = None
        should_keep = False

        # Buffer properties.
        write_buffer = list()
        buff_max_size = 1e6

        for row in csv_reader:
            row_number = csv_reader.line_num

            # Print progress update.
            if row_number % 1e6 == 0:
                print(f'\t\tParsing row {row_number}')

            # Unpack row from CSV file.
            qdl_code, date, o, h, l, c, v, div, sp, o_adj, h_adj, l_adj, c_adj, v_adj = row

            # Only repeat meta table check when row symbol changes.
            if qdl_code != prev_qdl_code:
                prev_qdl_code = qdl_code

                # This check is expensive, so we only perform it when the 'Symbol' value changes
                # from row to row in the CSV file.
                should_keep = qdl_code in tickers

                if should_keep:
                    # Get foreign key for current symbol in meta table.
                    sid = tickers[qdl_code][0]

            if should_keep:
                # Append row record to write buffer.
                data_row = (sid, date, o, h, l, c, v, div, sp,
                            o_adj, h_adj, l_adj, c_adj, v_adj)
                write_buffer.append(data_row)

            # Write buffer to 'qdl_eod' database table, once buffer is full.
            if len(write_buffer) > buff_max_size:
                print(f'\t\tWrite buffer reached max size. Writing to database...')
                db.executemany('''
 INSERT INTO qdl_eod(
    symbol_id,
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
 ''', write_buffer)

                write_buffer.clear()

        if len(write_buffer) > 0:
            print(f'\t\tWriting remaining records to database...')
            db.executemany('''
INSERT INTO qdl_eod(
    symbol_id,
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
''', write_buffer)

            write_buffer.clear()

    shutil.rmtree(temp_dir)


def generate_meta_data(db: sqlite3.Connection):

    print('Generating metadata...')

    sid_symbol = db.execute('''
SELECT id, symbol FROM qdl_symbols;''').fetchall()

    for idx, (sid, symbol) in enumerate(sid_symbol):

        if idx % 200 == 0:
            print(f'Getting meta data for {idx} of {len(sid_symbol)}...')

        rows = db.execute('''
SELECT date, adj_open, adj_close FROM qdl_eod WHERE symbol_id=? ORDER BY date;''',
                          (sid,)).fetchall()

        if len(rows) < 2:
            print(f'{symbol} missing data')
        else:
            first_date = datetime.strptime(rows[0][0], '%Y-%m-%d')
            last_date = datetime.strptime(rows[-1][0], '%Y-%m-%d')

            last_close = rows[-1][2]
            first_open = rows[0][1]

            returns = (last_close - first_open) / first_open
            returns /= (last_date - first_date).days
            returns *= 365

            days = (last_date - first_date).days

            db.execute('''
    UPDATE qdl_symbols
    SET start_date=?, end_date=?, lifetime_returns=?
    WHERE id=?;''', (first_date, last_date, returns, sid))

            db.commit()

    print('Done')


def purge_empty(db: sqlite3.Connection):

    meta = db.execute('''
SELECT id, symbol, start_date, end_date FROM qdl_symbols;''')

    for sid, symbol, start_date, end_date in meta:
        if start_date is None or end_date is None:
            print(f'Purging {symbol} due to empty data')

            db.execute('''
DELETE FROM qdl_eod WHERE symbol_id=?;
''', (sid,))

            db.execute('''
DELETE FROM qdl_symbols WHERE id=?;
''', (sid,))

    db.commit()


def main():

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
        prepare_database(db)
        get_quandl_tickers(db)
        bulk_download(db)
        generate_meta_data(db)
        purge_empty(db)


main()
