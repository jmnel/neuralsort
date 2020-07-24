from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.client import MarketDataTypeEnum
from ibapi.ticktype import TickTypeEnum

import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 68 and reqId == 1:
            print(f'last price: {price}')

    def tickSize(self, reqId, tickType, size):
        print(f'size: {size}')


def run_loop():
    app.run()


app = IBapi()
app.connect('127.0.0.1', 4002, 1)

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)

googl_contract = Contract()
googl_contract.symbol = 'GOOGLSASD'
googl_contract.secType = 'STK'
googl_contract.exchange = 'SMART'
googl_contract.currency = 'USD'

app.reqMarketDataType(MarketDataTypeEnum.DELAYED)
app.reqMktData(1, googl_contract, '', False, False, [])

time.sleep(10)
# app.disconnect()
