import sqlite3
from pprint import pprint

import settings

db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME)

days = tuple(zip(*db.execute('SELECT * FROM iex_days WHERE date >= ? and date <= ?;',
                             ('2019-01-01', '2019-12-31')).fetchall()))[0]

pprint(days)

for day_idx, day in enumerate(days):

    print(f'Counting messages in day {day_idx+1} of {len(days)}.')

    rows = db.execute('''
SELECT symbol, COUNT(timestamp)
FROM iex_trade_reports 
WHERE day=?
GROUP BY symbol;''', (day,)).fetchall()

    pprint(rows)
    exit()


db.close()
