import os
os.sys.path.insert(0,os.getcwd())

from pathlib import Path
assert Path().cwd().name == 'src'

from datetime import datetime, date, timedelta
from enum import Enum
from operator import itemgetter
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict
import csv
import io
import json
import re
import requests
import shutil
import sqlite3
import tempfile
import zipfile

import pandas as pd
import pandas_market_calendars as mcal
import quandl

import settings
from ticker_filter import TickerFilter
from data_import.vix_import import import_vix
from data_import.sp500_import import import_sp500

QUANDL_DATABASE_PATH = settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME
INFO_PATH = settings.DATA_DIRECTORY / settings.QUANDL_IMPORT_INFO_FILE


class UpdateType(Enum):
    NONE = 0
    PARTIAL = 1
    COMPLETE = 2


def determine_update_type() -> UpdateType:

    # Check if database file exists.
    if not QUANDL_DATABASE_PATH.is_file():
        print(f'Database {QUANDL_DATABASE_PATH} does not exist; needs rebuild.')
        return UpdateType.COMPLETE, {'log': list()}

    # Check for info json file.
    if not INFO_PATH.is_file():
        print(f'{INFO_PATH} not found. Did you remember to run migrate.py first.')
        return UpdateType.NONE, None

    # Try to parse and validate info json file.
    try:
        with open(INFO_PATH) as info_file:
            info = json.loads(info_file.read())

    except JSONDecodeError as e:
        # Decoding json failed.
        print(f'{INFO_PATH} is corrupted. JSON decode error.')
        INFO_PATH.unlink()
        return UpdateType.COMPLETE, {'log': list()}

    else:
        # Decoding json succeeded; validate the file.
        for entry in info:
            EXPECTED_KEYS = {'date',
                             'last_refresh_date',
                             'size',
                             'num_symbols',
                             'num_days',
                             'version',
                             'type'}
            if set(entry.keys()) != EXPECTED_KEYS:
                print(f'{INFO_PATH} is corrupted. Bad keys.')
                INFO_PATH.unlink()
                return UpdateType.COMPLETE, {'log': list()}

        # Check for existing entries.
        if len(info) < 1:
            print(f'No past updates.')
            return UpdateType.COMPLETE, {'log': list()}

    # Check for version bump.
    assert len(info) > 0
    if info[-1]['version'] != settings.QUANDL_DATABASE_VERSION:
        print(f'Database is using old version {info[-1]["version"]}.')
        print(f'New version is {settings.QUANDL_DATABASE_VERSION}. Needs to rebuild.')
        return UpdateType.COMPLETE, {'log': info}

    # Find last date from database.
    print('Finding last update date from database.')
    db = sqlite3.connect(QUANDL_DATABASE_PATH)
    old_num_days = db.execute('SELECT COUNT(DISTINCT date) from qdl_eod;').fetchall()[0][0]
    last_date = db.execute('SELECT MAX(date) from qdl_eod;').fetchall()[0][0]
    db.close()

    # Find last date from json info file.
    last_date_log = info[-1]['date']

    assert last_date == last_date_log

    print(f'Downloading partial EOD update from Quandl.')
    quandl.bulkdownload('EOD',
                        api_key=settings.QUANDL_API_KEY,
                        download_type='partial')
    temp_dir = Path(tempfile.mkdtemp(prefix="quandl"))
    shutil.move(str(Path.cwd() / 'EOD.partial.zip'), temp_dir)
    zip_path = temp_dir / 'EOD.partial.zip'

    partial_size = zip_path.stat().st_size

    print(f'Unziping {zip_path}.')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    #csv_path = Path(temp_dir) / os.listdir(temp_dir)[-1]
    csv_path = Path(temp_dir) / filter(lambda x: x.suffix == '.csv', os.listdir(temp_dir))[0]
    assert csv_path.suffix == '.csv'

    with open(csv_path, newline='') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = list(csv_reader)

    shutil.rmtree(temp_dir)

    assert len(rows[0]) == 14

    rows = [(r[0],
             datetime.strptime(r[1], '%Y-%m-%d').date(),
             float(r[2]),   # open
             float(r[3]),   # high
             float(r[4]),   # low
             float(r[5]),   # close
             float(r[6]),   # volume
             float(r[7]),   # divident
             float(r[8]),   # split
             float(r[9]),   # adj. open
             float(r[10]),  # adj. high
             float(r[11]),  # adj. low
             float(r[12]),  # adj. close
             float(r[13]),  # adj. volume
             ) for r in rows]

    rows.sort(key=itemgetter(1))

    newest_date = rows[-1][1].strftime('%Y-%m-%d')
#    last_date = '2020-07-13'

    print(f'Database last updated: {last_date}')
    print(f'Lastest available: {newest_date}')

    # Get NYSE and NASDAQ trading days in current date range.
    nyse = mcal.get_calendar('NYSE').schedule(start_date=last_date, end_date=newest_date)
    nasdaq = mcal.get_calendar('NASDAQ').schedule(start_date=last_date, end_date=newest_date)

    # Check assumption that NYSE and NASDAQ have same calendars.
    pd.testing.assert_frame_equal(nyse, nasdaq)

    missing_days = len(nyse.index) - 1

    if missing_days < 1:
        print('Database is up-to-date.')
        return UpdateType.NONE, {'log': info}

    if missing_days == 1:
        print('Database is behind 1 day. Can patch.')
        return UpdateType.PARTIAL, {'log': info,
                                    'patch': rows,
                                    'new_date': newest_date,
                                    'size': partial_size,
                                    'num_days': old_num_days + 1}

    else:
        print(f'Database is behind {missing_days} days; need to rebuild.')
        return UpdateType.COMPLETE, {'log': info}


def init_database():

    db = sqlite3.connect(QUANDL_DATABASE_PATH)

    print('Preparing qdl.sqlite3 database.')
    db.execute('PRAGMA foreign_keys = ON;')

    print('Creating qdl_symbol table.')
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

    print('Creating qdl_eod table.')
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

    # Create symbol EOD index.
    print('Creating qdl_eod_symbol_index index.')
    db.execute('''
CREATE INDEX qdl_eod_symbol_index ON qdl_eod(symbol);
''')


def delete_database():
    if QUANDL_DATABASE_PATH.is_file():
        print(f'Deleting {QUANDL_DATABASE_PATH}.')
        QUANDL_DATABASE_PATH.unlink()


def get_quandl_metadata():

    db = sqlite3.connect(QUANDL_DATABASE_PATH)
    if settings.QUANDL_API_KEY == None or settings.QUANDL_API_KEY == '':
        print('Quandl API key not set. Refer to settings.py')
        raise ValueError('Quandl API key not set. Refer to settings.py')

    print('Downloading Quandl EOD metadata.')

    META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'
    response = requests.get(META_ENDPOINT.format(settings.QUANDL_API_KEY))
    if not response.ok:
        print(f'Quandl metadata endpoint request error {response.status_code}.')
        exit()

    exchange_re = re.compile(r'<p><b>Exchange<\/b>: ([\w ]+)<\/p>')

    ss = io.BytesIO(response.content)
    zf = zipfile.ZipFile(ss)

    print('Processing metadata.')
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

    print(f'Found {count} symbols, filtered to {len(symbol_list)}')
    print('Done')

    # Insert list of symbols into meta table.
    db.executemany('''
INSERT OR REPLACE INTO qdl_symbols(symbol, qdl_code, name, exchange, is_common)
VALUES(?, ?, ?, ?, ?);
''', symbol_list)

    db.commit()
    db.close()


def patch_database(data):
    print('Performing database patch.')

    rows = data['patch']
    print(f'pre filter: {len(rows)}')

    get_quandl_metadata()

    db = sqlite3.connect(QUANDL_DATABASE_PATH)
    rows2 = db.execute('SELECT qdl_code FROM qdl_symbols;').fetchall()
    symbols = set(r[0] for r in rows2)

    rows = list(filter(lambda r: r[0] in symbols, rows))

    print(f'after: {len(rows)}')

    new_date = datetime.strptime(data['new_date'], '%Y-%m-%d').date()

    new_rows = list(filter(lambda r: r[1] == new_date, rows))
    update_rows = list(filter(lambda r: r[1] != new_date, rows))

    update_search = list((r[0], r[1].strftime('%Y-%m-%d')) for r in update_rows)
    rows = list((r[0], r[1].strftime('%Y-%m-%d'), *r[2:]) for r in rows)

    print(f'Inserting {len(new_rows)} new rows.')
    print(f'Updating {len(update_rows)} old rows.')

    # Delete old values.
    print(f'Deleting old rows.')
    db.executemany('''
DELETE FROM qdl_eod WHERE symbol=? AND date=?;''',
                   update_search)

    # Insert new and updated values.
    print(f'Inserting new rows.')
    db.executemany('''
 INSERT INTO qdl_eod(
    symbol, date, open, high, low, close, volume, dividend, split,
    adj_open, adj_high, adj_low, adj_close, adj_volume)
 VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);''',
                   rows)

    db.commit()
    db.close()

    new_info = {'date': data['new_date'],
                'last_refresh_date': datetime.now().timestamp(),
                'num_symbols': len(symbols),
                'num_days': data['num_days'],
                'size': data['size'],
                'type': 'partial',
                'version': settings.QUANDL_DATABASE_VERSION}

    info = data['log']
    info.append(new_info)

    with open(INFO_PATH, 'w') as info_file:
        info_file.write(json.dumps(info, indent=4, sort_keys=True))


def rebuild_database(data):

    # Delete old database.
    delete_database()
    init_database()

    # Import metadata.
    get_quandl_metadata()

    # Perform bulk download.
    print('Downloading bulk EOD data file.')

    quandl.bulkdownload('EOD',
                        api_key=settings.QUANDL_API_KEY,
                        download_type='complete')

    # Move to temp. directory.
    temp_dir = Path(tempfile.mkdtemp(prefix="quandl"))

    zip_path = temp_dir / 'EOD.zip'
    shutil.move(Path().cwd() / 'EOD.zip', zip_path)

    print('Download done.')

    # Create temporary directory.
    print('Extracting zip file.')

    # Extract zip file.
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    download_size = zip_path.stat().st_size
    zip_path.unlink()

    print('Parsing CSV file.')

    # Get path to csv file in temporary directory.
    csv_path = Path(temp_dir).absolute() / os.listdir(temp_dir)[0]
    assert csv_path.suffix == '.csv'

    db = sqlite3.connect(QUANDL_DATABASE_PATH)

    # Get list of symbols to keep.
    tickers = db.execute('''
SELECT symbol, qdl_code FROM qdl_symbols;
''').fetchall()

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
                print(f'Parsing row {row_number}')

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
                print(f'Write buffer reached max size. Writing to database.')
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
            print(f'Writing remaining records to database.')
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

    num_symbols = len(tickers)

    num_days = db.execute('SELECT COUNT(DISTINCT date) FROM qdl_eod;').fetchall()[0][0]
    new_date = db.execute('SELECT MAX(date) FROM qdl_eod;').fetchall()[0][0]

    db.close()

    new_info = {'date': new_date,
                'last_refresh_date': datetime.now().timestamp(),
                'num_symbols': num_symbols,
                'num_days': num_days,
                'size': download_size,
                'type': 'complete',
                'version': settings.QUANDL_DATABASE_VERSION}

    info = data['log']
    info.append(new_info)

    with open(INFO_PATH, 'w') as info_file:
        info_file.write(json.dumps(info, indent=4, sort_keys=True))


def purge_empty():
    print('Purging symbols that have no data.')

    db = sqlite3.connect(QUANDL_DATABASE_PATH)
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
    db.close()

    print('Done.')


def generate_meta_data():

    print('Generating metadata.')

    db = sqlite3.connect(QUANDL_DATABASE_PATH)

    symbols = tuple(zip(*db.execute('''
SELECT symbol FROM qdl_symbols;''').fetchall()))[0]

    for idx, symbol in enumerate(symbols):

        if idx % 200 == 0:
            print(f'Getting meta data for {idx} of {len(symbols)}.')

        rows = db.execute('''
SELECT date, adj_open, adj_close FROM qdl_eod WHERE symbol=? ORDER BY date;''',
                          (symbol,)).fetchall()

        if len(rows) < 2:
            print(f'Symbol {symbol} is missing data.')

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

    db.close()


def main():
    """
    Program entry point.

    """

    update_type, data = determine_update_type()

    if update_type == UpdateType.NONE:
        print('Database can not be updated.')
        exit()

    elif update_type == UpdateType.PARTIAL:
        patch_database(data)
        generate_meta_data()
        purge_empty()
        import_vix()
        import_sp500()

    elif update_type == UpdateType.COMPLETE:
        rebuild_database(data)
        generate_meta_data()
        purge_empty()
        import_vix()
        import_sp500()


if __name__ == '__main__':
    main()
