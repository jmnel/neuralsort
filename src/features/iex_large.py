import sqlite3

import settings

IEX_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

db = sqlite3.connect(IEX_PATH)

db.execute('DROP INDEX iex_trade_reports_2000_idx;')
db.execute('DROP TABLE iex_trade_reports_2000;')
db.execute('DROP TABLE iex_meta_2000;')

db.execute('''
CREATE TABLE iex_trade_reports_2000(
    id INTEGER PRIMARY KEY,
    day DATE,
    timestamp UNSIGNED BIG INT,
    symbol CHAR(16),
    price FLOAT,
    size FLOAT);''')
db.execute('''
CREATE TABLE iex_meta_2000(
    id INTEGER PRIMARY KEY,
    day DATE,
    symbol CHAR(16),
    message_count INTEGER);''')
db.execute('CREATE INDEX iex_trade_reports_2000_idx ON iex_trade_reports_2000(day, symbol);')

rows = db.execute('''
SELECT day, symbol, message_count FROM iex_ticks_meta
WHERE message_count >= 2000;''').fetchall()

rows = rows[:10000]

buff = list()

for idx, (day, symbol, message_count) in enumerate(rows):
    db.execute('''
INSERT INTO iex_meta_2000(day, symbol, message_count)
VALUES(?, ?, ?);''', (day, symbol, message_count))

    x = db.execute('''
SELECT day, symbol, timestamp, symbol, price, size FROM iex_trade_reports
WHERE day=? AND symbol=?
ORDER BY timestamp;''', (day, symbol)).fetchall()

#    db.executemany('''
# INSERT INTO iex_trade_reports_2000(day, symbol, timestamp, symbol, price, size)
# VALUES(?, ?, ?, ?, ?, ?);''', x)

#    db.commit()

    buff.extend(x)

    if len(buff) > 1e7:
        db.executemany('''
    INSERT INTO iex_trade_reports_2000(day, symbol, timestamp, symbol, price, size)
    VALUES(?, ?, ?, ?, ?, ?);''', buff)
        db.commit()
        buff.clear()

    if idx % 100 == 0:
        print(f'Wrote {idx+1} of {len(rows)}')

if len(buff) > 1e7:
    db.executemany('''
INSERT INTO iex_trade_reports_2000(day, symbol, timestamp, symbol, price, size)
VALUES(?, ?, ?, ?, ?, ?);''', buff)
    db.commit()
    buff.clear()

db.close()
