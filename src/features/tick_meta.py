import sqlite3
from pprint import pprint

import settings

IEX_DB_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

db = sqlite3.connect(IEX_DB_PATH)

db.execute('DROP TABLE IF EXISTS iex_ticks_meta;')
db.execute('''
CREATE TABLE IF NOT EXISTS iex_ticks_meta(
    id INTEGER PRIMARY KEY,
    day DATE NOT NULL,
    symbol CHAR(16) NOT NULL,
    message_count INTEGER DEFAULT(0));
    ''')

days = tuple(zip(*db.execute('SELECT date FROM iex_days;').fetchall()))[0]

for idx, day in enumerate(days):

    print(f'Enumerating symbols for day -> {day}')

    rows = db.execute('''
SELECT symbol, COUNT(id) FROM iex_trade_reports
WHERE day=? GROUP BY symbol ORDER BY symbol;''', (day,)).fetchall()
    rows = tuple((day, *r) for r in rows)

    db.executemany('''
INSERT INTO iex_ticks_meta(day, symbol, message_count)
VALUES(?, ?, ?);''',
                   rows)

    db.execute('UPDATE iex_days SET message_count=? WHERE date=?', (len(rows), day))
    db.commit()


db.close()
