import sqlite3
from pprint import pprint

import settings

IEX_DB_PATH = settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME

db = sqlite3.connect(IEX_DB_PATH)

days = tuple(zip(*db.execute('SELECT date FROM iex_days;').fetchall()))[0]

pprint(days)

db.close()
