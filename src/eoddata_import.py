import shutil
import os
import zipfile
from tempfile import mkdtemp
from datetime import datetime
import re
import csv
from pathlib import Path
import numpy as np
import sqlite3

import settings

eod_data_directory = settings.DATA_DIRECTORY / 'eoddata'


def prepare_database(db: sqlite3.Connection):

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('''
DROP INDEX IF EXISTS edodata_eod_symbols_index;
''')

    db.execute('''
DROP TABLE IF EXISTS eoddata_eod;
''')

    db.execute('''
DROP TABLE IF EXISTS eoddata_symbols;
''')

    db.execute('''
CREATE TABLE IF NOT EXISTS eoddata_symbols(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL
);''')

    db.execute('''
CREATE TABLE IF NOT EXISTS eoddata_eod(
    id INTEGER PRIMARY KEY,
    symbol_id MEDIUMINT UNSIGNED NOT NULL,
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    FOREIGN KEY(symbol_id) REFERENCES eoddata_symbols(id)
);''')

    db.execute('''
CREATE INDEX eoddata_eod_symbols_index ON eoddata_eod(symbol_id);
''')


def unzip_files():

    print('Unzipping files')

    temp_dir = mkdtemp(prefix='eoddata_')

    for zip_name in os.listdir(eod_data_directory):
        if zip_name.endswith('.zip'):

            zip_name = eod_data_directory / zip_name

            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

    print('Done\n')

    return temp_dir


def parse_csv(db, temp_dir):

    print('Parsing csv files')

    data = dict()

    for idx, csv_name in enumerate(os.listdir(temp_dir)):

        csv_name = Path(temp_dir) / csv_name

        symbols = dict()

        with open(csv_name, newline='') as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=' ')

            header = next(csv_reader)

            for row in csv_reader:
                symbol, date, o, h, l, c, v = row[0].split(',')

                if symbol in data.keys():
                    symbols[symbol] = symbols[symbol]
                else:

            print(header)
            return

    print('Done\n')


def cleanup(temp_dir: Path):

    print('Cleaning up...')
    shutil.rmtree(temp_dir)
    print('Done\n')


def main():

    with sqlite3.connect(eod_data_directory / 'eoddata.db') as db:

        prepare_database(db)
        temp_dir = unzip_files()
        parse_csv(db, temp_dir)
        cleanup(temp_dir)


main()
