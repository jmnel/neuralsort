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
import logging

from ticker_filter import TickerFilter
import settings


def prepare_database(db: sqlite3.Connection):

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('''
DROP TABLE IF EXISTS qdl_eod;
''')

    db.execute('''
DROP TABLE IF EXISTS qdl_symbols;
''')

    # Create meta table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_symbols(
    symbol CHAR(32) PRIMARY KEY,
    qdl_code CHAR(32) NOT NULL,
    name CHAR(256) NOT NULL,
    exchange CHAR(16) NOT NULL,
    last_trade DATE,
    start_date DATE,
    end_date DATE,
    lifetime_returns FLOAT,
    is_common BOOLEAN
    );
''')

    # Create data table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_eod(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL,
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
    FOREIGN KEY(symbol) REFERENCES qdl_symbols(symbol)
);''')


logger = logging.getLogger(__name__)


def get_quandl_tickers(db: sqlite3.Connection):

    logger.info('Downloading Quandl EOD metadata.')

    META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'
    response = requests.get(META_ENDPOINT.format(settings.QUANDL_API_KEY))
    if not response.ok:
        logger.critical(f'Quandl metadata endpoint request error {response.status_code}.')
        exit()

    exchange_re = re.compile(r'<p><b>Exchange<\/b>: ([\w ]+)<\/p>')

    ss = io.BytesIO(response.content)
    zf = zipfile.ZipFile(ss)

    logger.info('Processing metadata.')
    count = 0
    with zf.open('EOD_metadata.csv', 'r') as f:
        f2 = io.TextIOWrapper(f, encoding='utf-8')
        reader = csv.reader(f2, delimiter=',')
        next(reader)

        symbol_list = list()
        for idx, row in enumerate(list(reader)):

            count += 1

            symbol = row[0]
            name = row[1][:-38 - len(row[0])]
            qdl_code = row[0]
#            qdl_code = 'EOD/' + row[0]

            # Extract exchange from description column.
            m = exchange_re.search(row[2])
            if m:
                exchange = m.groups()[0]

            # Symbol AA_P has no description, exchange, or data.
            else:
                continue

            symbol_list.append((symbol, qdl_code, name, exchange, True))

    # Filter out non-common stocks.
    symbol_list = list(
        filter(lambda row: TickerFilter.filter(row[0], row[2], row[3]), symbol_list))

    logger.info(f'Found {count} symbols, filtered to {len(symbol_list)}')
    logger.info('Done')

    # Insert list of symbols into meta table.
    db.executemany('''
INSERT INTO qdl_symbols(symbol, qdl_code, name, exchange, is_common)
VALUES(?, ?, ?, ?, ?);
''', symbol_list)

    db.commit()


def bulk_download(db: sqlite3.Connection):

    REDOWNLOAD = False

    eod_path = settings.DATA_DIRECTORY / 'quandl' / 'EOD.zip'

    # Download file if it is missing or flag is set.
    if REDOWNLOAD or not eod_path.is_file():
        logger.info('\tDownloading bulk EOD data file.')

        quandl.bulkdownload('EOD',
                            api_key=settings.QUANDL_API_KEY,
                            download_type='complete')

        shutil.move('EOD.zip', eod_path)

        logger.info('Done.')
    else:
        logger.warning('Already downloaded.')

    # Create temporary directory.
    logger.info('Extracting zip file.')
    temp_dir = Path(mkdtemp(prefix='quandl_eod'))

    # Extract zip file.
    with zipfile.ZipFile(eod_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    logger.info('Parsing CSV file.')

    # Get path to csv file in temporary directory.
    csv_name = os.listdir(temp_dir)[0]
    csv_path = Path(temp_dir).absolute() / csv_name

    # Get list of symbols to keep.
    tickers = db.execute('''
SELECT symbol, qdl_code FROM qdl_symbols;
''').fetchall()

#    tickers = {row[1].split('/')[1]: (row[1], row[0]) for row in tickers}
#    tickers = {row[1].split('/')[1]: row[0] for row in tickers}

    tickers = set(t[0] for t in tickers)

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

            assert len(qdl_code) > 0

            if qdl_code[0:4] == 'EOD/':
                qdl_code = qdl_code[4:]
            assert qdl_code[0:4] != 'EOD/'

            # Only repeat meta table check when row symbol changes.
            if qdl_code != prev_qdl_code:
                prev_qdl_code = qdl_code

                # This check is expensive, so we only perform it when the 'Symbol' value changes
                # from row to row in the CSV file.
                should_keep = qdl_code in tickers

            if should_keep:
                # Append row record to write buffer.
                data_row = (qdl_code, date, o, h, l, c, v, div, sp,
                            o_adj, h_adj, l_adj, c_adj, v_adj)
                write_buffer.append(data_row)

            # Write buffer to 'qdl_eod' database table, once buffer is full.
            if len(write_buffer) > buff_max_size and len(write_buffer) > 0:
                print(f'\t\tWrite buffer reached max size. Writing to database...')
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
 ''', write_buffer)
                db.commit()

                write_buffer.clear()

        if len(write_buffer) > 0:
            print(f'\t\tWriting remaining records to database...')
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
''', write_buffer)
            db.commit()

            write_buffer.clear()

    shutil.rmtree(temp_dir)


def generate_meta_data(db: sqlite3.Connection):

    logger.info('Generating metadata.')

    symbols = tuple(zip(*db.execute('''
SELECT symbol FROM qdl_symbols;''').fetchall()))[0]

    for idx, symbol in enumerate(symbols):

        print(f'{idx} : {symbol}')

        if idx % 200 == 0:
            logger.info(f'Getting meta data for {idx} of {len(symbols)}.')

        rows = db.execute('''
SELECT date, adj_open, adj_close FROM qdl_eod WHERE symbol=? ORDER BY date;''',
                          (symbol,)).fetchall()

        if len(rows) < 2:
            logger.warning(f'Symbol {symbol} is missing data.')

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
    WHERE symbol=?;''', (first_date, last_date, returns, symbol))

            db.commit()

    logger.info('Done.')


def purge_empty(db: sqlite3.Connection):

    logger.info('Purging symbols that have no data.')

    meta = db.execute('''
SELECT symbol, start_date, end_date FROM qdl_symbols;''')

    for symbol, start_date, end_date in meta:
        if start_date is None or end_date is None:
            print(f'Purging {symbol} due to empty data')

            db.execute('''
DELETE FROM qdl_eod WHERE symbol=?;
''', (symbol,))

            db.execute('''
DELETE FROM qdl_symbols WHERE symbol=?;
''', (symbol,))

    db.commit()

    logger.info('Done.')


def main():

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
        #        prepare_database(db)
        #        get_quandl_tickers(db)
        #        bulk_download(db)
        generate_meta_data(db)
        purge_empty(db)


main()
