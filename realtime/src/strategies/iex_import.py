import csv
import sqlite3
from pprint import pprint
from pathlib import Path
import os

import settings


def prepare_database() -> sqlite3.Connection:

    db = sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME)

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('DROP TABLE IF EXISTS iex_trade_reports;')
    db.execute('''
CREATE TABLE iex_trade_reports(
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    timestamp UNSIGNED BIG INT NOT NULL,
    symbol CHAR(16) NOT NULL,
    price INTEGER NOT NULL,
    size INTEGER NOT NULL);
''')

    return db


def main():

    db = prepare_database()

    csv_directory = settings.DATA_DIRECTORY / 'csv'

    csv_path = csv_directory / '20200609.csv'
    with open(csv_path, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        date_str = '-'.join((f[:4], f[4:6], f[6:8]))

    db.close()


#    for f in os.listdir(csv_directory):
#        csv_path = csv_directory / f
#        date_str = '-'.join((f[:4], f[4:6], f[6:8]))

#        with open(csv_path, 'r') as csv_file:
#            rows = csv.reader(csv_file, delimiter=',')
#            for idx in range(3):
#                print(next(rows))
#            print()


if __name__ == '__main__':
    main()
