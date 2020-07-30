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

    def historicalTicksLast(self, reqId, ticks, done):
        for tick in ticks:
            print('tick: {reqId}: {tick}')


def run_loop():
    app.run()


# Create IB client and connect to IB Gateway API.
app = IBapi()
app.connect('127.0.0.1', 4002, 2)

# Launch client thread.
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Give client time to establish conneciton.
time.sleep(1)

googl_contract = Contract()
googl_contract.symbol = 'GOOGL'
googl_contract.secType = 'STK'
googl_contract.exchange = 'SMART'
googl_contract.currency = 'USD'
app.reqHistoricalTicks(1,
                       googl_contract,
                       "20170712 21:39:33",
                       "",
                       10,
                       "TRADES",
                       1,
                       True,
                       [])

time.sleep(10)

app.disconnect()
