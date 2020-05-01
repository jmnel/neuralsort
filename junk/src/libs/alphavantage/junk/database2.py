import os.path
import sqlite3


class Database:

    def __init__(self):
        pass

    def add_symbol(self, symbol: str):
        # Check if meta data table exists.
        sql = \
            '''CREATE TABLE IF NOT EXISTS symbols_meta (
                id integer primary_key,
                info char(64) not null,
                symbol char(16) not null,
                last_refreshed date not null,
                output_size char(16) not null,
                time_zone char(32),
                data_table char(16)
                );'''

        conn = self._connect()
        assert(conn is not None)
        try:
            c = conn.cursor()
            res = c.execute(sql)
            foo = res.fetchall()
#            print(foo)
        except sqlite3.Error as e:
            print(e)

    def _connect(self):
        connection = None
        try:
            connection = sqlite3.connect('test.db')
            return connection
        except sqlite3.Error as e:
            print(e)

        return connection


db = Database()
db.add_symbol('FOO')
