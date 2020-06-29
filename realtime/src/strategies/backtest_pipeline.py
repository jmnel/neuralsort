import sqlite3
from pprint import pprint

import settings


class BacktestPipeline:

    def __init__(self):

        iex_db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME)

        day_end = '2019-12-31'
        days = iex_db.execute('SELECT date FROM iex_days WHERE date <= ?;',
                              (day_end,)).fetchall()
        days = tuple(d[0] for d in days)

        pprint(days)


foo = BacktestPipeline()
