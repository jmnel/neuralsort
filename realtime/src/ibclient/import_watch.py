"""
Imports a list of symbols to watch for a given day. Symbols are provided as a csv file.

"""

import sqlite3
import csv
from pprint import pprint
from datetime import datetime
import os
from pathlib import Path

import settings

IB_DB_PATH = settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME

db = sqlite3.connect(IB_DB_PATH)

# Enable foreign keys in database.
db.execute('PRAGMA foreign_keys = ON;')

# Create 'ib_watch' table if it does not exist.
db.execute('''
CREATE TABLE IF NOT EXISTS ib_watch(
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    type CHAR(16),
    primary_exchange CHAR(32),
    currency CHAR(8));''')

# Create 'ib_days' table if it does not exist.
db.execute('''
CREATE TABLE IF NOT EXISTS ib_days(
    date DATE PRIMARY KEY,
    message_count INTEGER DEFAULT 0);
    ''')

# Create 'ib_trade_reports' table if it does not exist.
db.execute('''
CREATE TABLE IF NOT EXISTS ib_trade_reports(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    timestamp UNSIGNED BIG INT NOT NULL,
    symbol CHAR(16) NOT NULL,
    price INTEGER NOT NULL,
    size INTEGER NOT NULL,
    FOREIGN KEY (day) REFERENCES ib_days(date));''')

# Loop through all csv files in watch directory.
for csv_file in os.listdir(settings.DATA_DIRECTORY / 'watch'):
    csv_path = settings.DATA_DIRECTORY / 'watch' / csv_file

    # Check if file type is csv.
    if csv_path.suffix == '.csv':
        with open(csv_path, 'r') as csv_data:
            rows = tuple(csv.reader(csv_data))

        # Get date from filename stem.
        date = Path(csv_file).stem

        # Insert day into database.
        db.execute('INSERT OR IGNORE INTO ib_days(date) VALUES(?);',
                   (date,))
        rows = tuple((date, r[0]) for r in rows)

        # Insert symbols into 'ib_watch' table.
        db.executemany('''
INSERT OR IGNORE INTO ib_watch(
    date, symbol) VALUES(?, ?);''', rows)

db.commit()


db.close()
