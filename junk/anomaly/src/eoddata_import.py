import os
from tempfile import mkdtemp
from pathlib import Path
import shutil
import zipfile
import csv
import sqlite3
from datetime import datetime
import re

import settings


def unzip_files():

    print('Unzipping files')
    eod_directory = settings.DATA_DIRECTORY / 'eoddata'

    temp_dir = Path(mkdtemp(prefix='eoddata'))

    for zip_name in os.listdir(eod_directory):
        if zip_name.endswith('.zip'):
            zip_name = eod_directory / zip_name
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

    print('Done\n')

    return temp_dir


def parse_csv(db, temp_dir):

    data = dict()

    print('Parsing CSV files')
    for idx, csv_name in enumerate(os.listdir(temp_dir)):

        if idx % 20 == 0:
            print(f'\tProcessing {idx} of {len(os.listdir(temp_dir))}...')

        exchange_name = str(csv_name).split('_', 2)[0]

        if exchange_name in data.keys():
            exchange_list = data[exchange_name]
        else:
            data[exchange_name] = dict()
            exchange_list = data[exchange_name]

        assert(exchange_name in settings.EXCHANGE_NAMES)

        csv_name = temp_dir / csv_name

        with open(csv_name, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')

            header = next(csv_reader)

            for row in csv_reader:
                symbol, date, o, h, l, c, v = row[0].split(',')

                if symbol in exchange_list.keys():
                    symbol_list = exchange_list[symbol]

                else:
                    exchange_list[symbol] = list()
                    symbol_list = exchange_list[symbol]

                date = datetime.strptime(date, '%d-%b-%Y')
                o = float(o)
                h = float(h)
                l = float(l)
                c = float(l)
                v = float(v)

                row = (date, o, h, l, c, v)

                symbol_list.append(row)

    print('\tDone\n')

    return data


def save_to_db(db, data):

    print('Saving to database')

    for idx, (exchange_name, symbols) in enumerate(data.items()):

        print('\tprocessing exchange {idx} of {len(data.items())}')

        db.execute('''
INSERT OR IGNORE INTO eod_exchanges(name) VALUES(?);
''', (exchange_name,))
        db.commit()

        exchange_id = db.execute('''
SELECT id FROM eod_exchanges WHERE name=?;
''', (exchange_name,)).fetchone()[0]

        print(exchange_id)

        for jdx, (symbol_name, rows) in enumerate(symbols.items()):

            if jdx % 100 == 0:
                print(f'\t\tprocessing symbol {jdx} of {len(symbols.items())}')

            db.execute('''
INSERT OR IGNORE INTO eod_symbols(symbol) VALUES(?);
''', (symbol_name,))
            db.commit()

            symbol_id = db.execute('''
SELECT id FROM eod_symbols WHERE symbol=?;
''', (symbol_name,)).fetchone()[0]

            db.execute('''
INSERT OR IGNORE INTO eod_exchange_symbol(symbol_id, exchange_id)
    VALUES(?, ?);
''', (symbol_id, exchange_id))
            db.commit()

            rows = [(symbol_id, exchange_id,
                     r[0], r[1],
                     r[2], r[3], r[4], r[5])
                    for r in rows]

            db.executemany('''
INSERT INTO eod_ohlcv(
    symbol_id, exchange_id, date,
    open, high, low, close, volume) VALUES(?, ?, ?, ?, ?, ?, ?, ?);
''', rows)

            db.commit()

        print('\t\tdone')

    print('\tdone')


def eod_temp_cleanup(temp_dir: Path):

    print('Cleaning up')
    shutil.rmtree(temp_dir)
    print('\tDone\n')

    return None


def prepare_database(db: sqlite3.Connection):

    print('Preparing database')

    db.execute('PRAGMA foreign_keys = ON;')
    rows = db.execute('PRAGMA foreign_keys;')

    cur = db.cursor()
    cur.execute('''
CREATE TABLE IF NOT EXISTS eod_exchanges(
    id INTEGER PRIMARY KEY,
    name CHAR(32) UNIQUE NOT NULL);
''')
    cur.execute('''
CREATE TABLE IF NOT EXISTS eod_symbols(
   id INTEGER PRIMARY KEY,
   symbol CHAR(10) UNIQUE NOT NULL);
 ''')

    cur.execute('''
CREATE TABLE IF NOT EXISTS eod_exchange_symbol(
   symbol_id MEDIUMINT UNSIGNED NOT NULL,
   exchange_id MEDIUMINT UNSIGNED NOT NULL,
   PRIMARY KEY(symbol_id, exchange_id),
   FOREIGN KEY(symbol_id) REFERENCES eod_symbols(id),
   FOREIGN KEY(exchange_id) REFERENCES eod_exchanges(id));
''')

    cur.execute('''
CREATE TABLE IF NOT EXISTS eod_ohlcv(
    id INTEGER PRIMARY KEY,
    symbol_id MEDIUMINT UNSIGNED NOT NULL,
    exchange_id MEDIUMINT UNSIGNED NOT NULL,
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    FOREIGN KEY(symbol_id) REFERENCES eod_symbols(id),
    FOREIGN KEY(exchange_id) REFERENCES eod_exchanges(id));
''')

    cur.execute('''
CREATE INDEX IF NOT EXISTS eod_ohlcv_symbols_index ON eod_ohlcv(symbol_id);
''')

    print('\tDone\n')
    return db


def clear_database(db):

    print('\tDeleting all records from database')

    db.execute('DELETE FROM eod_ohlcv;')
    db.execute('DELETE FROM eod_exchange_symbol;')
    db.execute('DELETE FROM eod_exchanges;')
    db.execute('DELETE FROM eod_symbols;')

    print('\tDone\n')


def delete_non_common_syms(db: sqlite3.Connection, pattern: re.Pattern):

    print('\tDeleting records for non common stocks...')
    symbols = db.execute('''
SELECT id, symbol from eod_symbols;
''').fetchall()

    non_common = list(filter(lambda id_sym: pattern.match(id_sym[1]), symbols))

    print(f'\t\t{len(non_common)} matches found out of {len(symbols)}')

    if not len(non_common):
        print('\t\tNone found\nDone\n')
        return

    remove_ids = list(sym[0] for sym in non_common)

    for idx, symbol_id in enumerate(remove_ids):
        if idx % 200 == 0:
            print(f'\t\t\tDeleting {idx} of {len(non_common)}')

        db.execute('DELETE FROM eod_ohlcv WHERE symbol_id=?;', (symbol_id,))
        db.execute(
            'DELETE FROM eod_exchange_symbol WHERE symbol_id=?;', (symbol_id,))
        db.execute('DELETE FROM eod_symbols WHERE id=?;', (symbol_id,))

    db.commit()

    print('\tDone\n')


def main():

    # Connect to database; create it if it doesn't exist already.
    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

        prepare_database(db)

        if settings.IMPORT_EOD_RAW:
            clear_database(db)
            temp_dir = unzip_files()
            data = parse_csv(db, temp_dir)

            temp_dir = eod_temp_cleanup(temp_dir)

            save_to_db(db, data)

        delete_non_common_syms(db, settings.EOD_NON_COMMON_PATTERN)


main()
