import settings
import logging
import sqlite3
from pathlib import Path
import json
from typing import Optional, Dict
from pprint import pprint
from datetime import datetime, date, timedelta
import pandas as pd
import pandas_market_calendars as mcal
import quandl
import shutil
import zipfile
import tempfile

logger = logging.getLogger(__name__)

QUANDL_DATABASE_PATH = settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME
INFO_PATH = settings.DATA_DIRECTORY / 'eod_import.json'


def delete_database(db_path: Path):
    logger.info(f'Deleting {db_path}.')
    if db_path.is_file():
        db_path.unlink()


def init_database(db: sqlite3.Connection):

    logger.info('Preparing qdl.sqlite3 database')

    # Enable foreign keys.
    db.execute('PRAGMA foreign_keys = ON;')

    # Create meta table.
    logger.info('Creating qdl_symbol table.')
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
    logger.info('Creating qdl_eod table.')
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

    # Create symbol EOD index; significantly sppeds up lookup by symbol.
    logger.info('Creating qdl_eod_symbol_index index.')
    db.execute('''
CREATE INDEX IF NOT EXISTS qdl_eod_symbol_index ON qdl_eod(symbol);
''')


def last_refresh_date(db: sqlite3.Connection):
    pass


def load_save_history(load_path: Path) -> Optional[Dict]:
    if not load_path.is_file():
        return None

    with open(load_path, 'rt') as conf_file:
        return json.loads(conf_file.read())


def dump_save_history(save_path: Path, conf: Dict):
    with open(save_path, 'w') as conf_file:
        conf_file.write(json.dumps(conf, indent=4, sort_keys=False))


# def missed_calendar_days(prev_date: date):

    #    if( datetime.now().hour > 7 ):


# def check_status():

    # Assume previous -2 days is no longer available at 7am.
#    if datetime.now().hour > 7:
#    n = datetime(2020, 7, 16, 8, 0)
#    if n.hour > 7:
#        print('more')
#        day_end = n.date()
#        day_end -= timedelta(days=2)
#        day_end = day_end.strftime('%Y-%m-%d')
#    else:
#        day_end = n.date()
#        day_end -= timedelta(days=3)
#        day_end = day_end.strftime('%Y-%m-%d')

#    print(day_end)

#    day_start = '2001-12-01'
#    day_end = '2020-02-01'
#    nasdaq = mcal.get_calendar('NASDAQ').schedule(start_date=day_start, end_date=day_end)
#    nyse = mcal.get_calendar('NYSE').schedule(start_date=day_start, end_date=day_end)
#    pd.testing.assert_frame_equal(nasdaq, nyse)

#    print(start_date)


def main():

    conf = [{'date': datetime(2020, 7, 14).timestamp(),
             'last_refresh_date': datetime.now().timestamp(),
             'size': 150,
             'num_symbols': 123,
             'version': '0.1'}]

    dump_save_history(INFO_PATH, conf)

    # Check if database rebuild json file exists.
    if not INFO_PATH.is_file():
        print(f'{INFO_PATH} not found.')
        return True, None

    # Try to parse info json file.
    try:
        with open(INFO_PATH) as info_file:
            info = json.loads(info_file.read())
    except json.JSONDecodeError as e:
        print(f'{INFO_PATH} corrupted.')
        return True, None

    # Check for entries in info.
    if len(info) < 1:
        print('No past recorded refreshes.')
        return True, None

    # Check for version bump.
    for entry in info:
        if entry['version'] != settings.QUANDL_DATABASE_VERSION:
            print(f'Version mismatch: {entry["version"]} -> {settings.QUANDL_DATABASE_VERSION}.')
            return True, None

    # Check if EOD database exists.
    if not QUANDL_DATABASE_PATH.is_file():
        print(f'Database {QUANDL_DATABASE_PATH} not found.')
        return True, None

    # Count missing market days.
    start_date = datetime.fromtimestamp(info[-1]['date']).date()

    quandl.bulkdownload('EOD',
                        api_key=settings.QUANDL_API_KEY,
                        download_type='partial')

    temp_dir = Path(tempfile.mkdtemp(prefix='quandl'))

    shutil.move(str(Path.cwd() / 'EOD.partial.zip'), temp_dir)

    return

    shutil.rmtree(temp_dir)

    return

    info = load_save_history(INFO_PATH)
#    if not info:
#        print('No

    missed_calendar_days(date(2020, 7, 12))

    db = sqlite3.connect(QUANDL_DATABASE_PATH)
    db.close()


# Program entry point.
if __name__ == '__main__':
    main()
