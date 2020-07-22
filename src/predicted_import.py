import sqlite3
import csv
from pathlib import Path
import os
from pprint import pprint

import settings

DB_PATH = settings.DATA_DIRECTORY / 'topk.sqlite3'

db = sqlite3.connect(DB_PATH)

db.execute('PRAGMA foreign_keys=on;')

db.execute('DROP TABLE IF EXISTS predicted;')
db.execute('DROP TABLE IF EXISTS true;')
db.execute('''
CREATE TABLE IF NOT EXISTS predicted(
        id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        symbol CHAR(16) NOT NULL);''')
db.execute('''
CREATE TABLE IF NOT EXISTS true(
        id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        symbol CHAR(16) NOT NULL);''')

# Import predicted top-k.
csv_dir = settings.DATA_DIRECTORY / 'predicted_topk'
csvs = list(filter(lambda x: Path(x).suffix == '.csv', os.listdir(csv_dir)))
csvs = sorted(csvs)

csv_path = csv_dir / csvs[-1]

with open(csv_path) as csv_file:
    reader = csv.reader(csv_file)

    records = list()
    for row in reader:
        date = row[0]
        symbols = row[1:]
        for s in symbols:
            records.append((date, s))

db.executemany('''
INSERT INTO predicted(date, symbol) VALUES( ?, ? );
''', records)

# Import true top-k.
csv_dir = settings.DATA_DIRECTORY / 'true_topk'
csvs = list(filter(lambda x: Path(x).suffix == '.csv', os.listdir(csv_dir)))
csvs = sorted(csvs)

csv_path = csv_dir / csvs[-1]

with open(csv_path) as csv_file:
    reader = csv.reader(csv_file)

    records = list()
    for row in reader:
        date = row[0]
        symbols = row[1:]
        for s in symbols:
            records.append((date, s))

db.executemany('''
INSERT INTO true(date, symbol) VALUES( ?, ? );
''', records)

db.commit()

db.close()
