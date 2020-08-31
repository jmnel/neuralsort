import sqlite3
from pprint import pprint
from time import perf_counter
from random import shuffle

import settings

IEX_DB_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

db = sqlite3.connect(IEX_DB_PATH)

rows = db.execute('SELECT * FROM iex_ticks_meta WHERE message_count LIMIT 100000;').fetchall()
rows = tuple(filter(lambda r: r[-1] >= 2000, rows))
# shuffle(rows)
pprint(rows)

t = perf_counter()

print(f'Fetching {len(rows)} series')

for _, day, symbol, count in rows:

    rows2 = db.execute('SELECT * FROM IEX_TRADE_REPORTS WHERE day=? AND symbol=?;',
                       (day, symbol)).fetchall()

print(f'Took {perf_counter() - t}')

db.close()
