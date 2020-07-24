"""
Check symbols in watch list to make sure they are tradable, and get Smart Routing information.

Only common stocks in USD currency are selected.

"""

import sqlite3
from pprint import pprint
import threading
import time
from datetime import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.client import MarketDataTypeEnum
from ibapi.ticktype import TickTypeEnum

import settings

results = list()


class IBapi(EWrapper, EClient):
    """
    Client that communicates with IB Gateway.

    """

    def __init__(self):
        EClient.__init__(self, self)

    def symbolSamples(self, reqId: int,
                      contractDescriptions):
        """
        Handle reponse to matching symbol lookup.

        Args:
            reqId:                      Request ID.
            contractDescriptions:       List of contract descriptions.

        """

        super().symbolSamples(reqId, contractDescriptions)

        candidiates = list()

        # Iterate through each item in contract description list.
        for desc in contractDescriptions:
            deriv_sec_types = ''
            candidiates.append((desc.contract.symbol,
                                desc.contract.secType,
                                desc.contract.primaryExchange,
                                desc.contract.currency))

        # Filter search result to only common stock, name matching query pattern exactly, and in USD currency.
        candidiates = list(filter(lambda desc: desc[0] == symbol and desc[1] == 'STK' and desc[3] == 'USD',
                                  candidiates))

        # Catch no match.
        assert len(candidiates) == 1

        # Store result.
        global results
        c = candidiates[0]
        results.append(c)


def run_loop():
    app.run()


# Create IB client and connect to IB Gateway API.
app = IBapi()
app.connect('127.0.0.1', 4002, 1)

# Launch client thread.
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Give client time to establish conneciton.
time.sleep(1)

# Specify day in watchlist to check.
#date = '2020-07-24'
date = datetime.now().strftime('%Y-%m-%d')

db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME)

# Get symbols on watchlist to check.
rows = db.execute('SELECT symbol FROM ib_watch WHERE date=?',
                  (date,)).fetchall()
rows = list([date, r[0], '', ''] for r in rows)

for idx in range(len(rows)):
    print(f'Querying {symbol}')

    # Make symbol match request to client.
    symbol = rows[idx][1]
    app.reqMatchingSymbols(idx, symbol)

    # Rate limit is 1 query per second.
    time.sleep(1.1)

# Give client time to finish requests.
time.sleep(10)

results = tuple((r[1], r[2], r[3], r[0]) for r in results)
pprint(results)

# Update 'ib_watch' table with query results.
db.executemany('''
UPDATE ib_watch SET type=?, primary_exchange=?, currency=?
    WHERE symbol==?;''', results)

db.commit()

db.close()
app.disconnect()
