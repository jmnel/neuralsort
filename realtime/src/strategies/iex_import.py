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
    db.execute('DROP TABLE IF EXISTS iex_trade_reports_days;')
    db.execute('DROP TABLE IF EXISTS iex_trade_reports_symbols;')
    db.execute('DROP TABLE IF EXISTS iex_trade_reports_day_symbols;')

    # Create days meta table.
#    db.execute('''
# CREATE TABLE iex_trade_reports_days(
#    id INTEGER PRIMARY KEY,
#    date DATE NOT NULL UNIQUE,
#    message_count UNSIGNED_INTEGER DEFAULT 0);
# ''')

    # Create symbols meta table.
#    db.execute('''
# CREATE TABLE iex_trade_reports_symbols(
#    id INTEGER PRIMARY KEY,
#    symbol CHAR(16) NOT NULL UNIQUE,
#    qdl_code CHAR(16),
#    name CHAR(256),
#    exchange CHAR(16));
# ''')

    # Create trade reports table.
#    db.execute('''
# CREATE TABLE iex_trade_reports(
#    id INTEGER PRIMARY KEY,
#    date DATE NOT NULL,
#    timestamp UNSIGNED BIG INT NOT NULL,
#    symbol CHAR(16) NOT NULL,
#    price INTEGER NOT NULL,
#    size INTEGER NOT NULL);
# ''')

    return db


def parse_csv(db):
    csv_directory = settings.DATA_DIRECTORY / 'csv'

    for f in os.listdir(csv_directory):
        csv_path = csv_directory / f
        if csv_path.is_file and csv_path.suffix == '.csv':
            with open(csv_path, 'r') as csv_file:
                rows = csv.reader(csv_file, delimiter=',')
                date_str = '-'.join((f[:4], f[4:6], f[6:8]))
                rows = list(rows)

                rows = list((date_str, r[0], r[1], r[2], r[3]) for r in rows)

#                pprint(rows[:3])
                db.executemany('''
INSERT INTO iex_trade_reports(date, timestamp, symbol, price, size)
VALUES(?, ?, ?, ?, ?);
''', rows)
                db.commit()


def main():
    db = prepare_database()
#    parse_csv(db)
    db.close()


if __name__ == '__main__':
    main()
