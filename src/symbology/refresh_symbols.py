from collections import namedtuple
from pprint import pprint
from pprint import pprint
import csv
import io
import re
import requests
import sqlite3
import zipfile
import threading
from time import sleep

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.client import MarketDataTypeEnum
from ibapi.ticktype import TickTypeEnum

import settings

QUANDL_META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'
NYSE_TEST_TICKERS = {'ATEST', 'CTEST', 'MTEST', 'NTEST', 'ZTST', 'CBX'}
NASDAQ_TEST_TICKERS = {'ZAZZT', 'ZBZZT', 'ZJZZT', 'ZVZZT', 'ZXYZ_A', 'ZVZZCNX'}
SYMBOLS_DB_PATH = settings.DATA_DIRECTORY / 'foo.sqlite3'
EXCHANGES_KEEP = {'NYSE', 'NASDAQ.SCM', 'NASDAQ.NMS', 'AMEX', 'BATS', 'ARCA'}
EXCHANGES_DISCARD = {'VALUE', 'PINK', 'PINK.NOINFO', 'PINK.CURRENT', 'LSE', 'LSEIOB1'}
EXCHANGES = EXCHANGES_KEEP.union(EXCHANGES_DISCARD)


def get_quandl_tickers():
    """
    Get list of all tickers in Quandl EOD dataset.

    """

    response = requests.get(QUANDL_META_ENDPOINT.format(settings.QUANDL_API_KEY))
    if not response.ok:
        print(f'Quandl metadata endpoint request error {response.status_code}')
        exit()

    ss = io.BytesIO(response.content)
    zf = zipfile.ZipFile(ss)

    with zf.open('EOD_metadata.csv', 'r') as f:
        f2 = io.TextIOWrapper(f, encoding='utf-8')
        reader = csv.reader(f2, delimiter=',')

        # Skip column header.
        next(reader)

        exchange_regex = re.compile(r'<p><b>Exchange</b>: (.*?)</p>')
        ticker_regex = re.compile(r'<p><b>Ticker</b>: (.*?)</p>')

        meta = list()
        Meta = namedtuple('Meta', 'qdl_code qdl_ticker qdl_exchange')

        for row in reader:
            qdl_code = row[0]

            # Filter out NYSE and NASDAQ test symbols.
            if qdl_code in NYSE_TEST_TICKERS or qdl_code in NASDAQ_TEST_TICKERS:
                continue

            # Skip symbols with trailing '_'. These appear to be convention for deprecated symbols.
            if qdl_code[-1] == '_':
                continue

            description = row[1][:-35 - len(qdl_code) - 3]
            m1 = exchange_regex.search(row[2])
            m2 = ticker_regex.search(row[2])

            if m1 and m2:
                exchange = m1.groups()[0]
                ticker = m2.groups()[0]

                meta.append(Meta(qdl_code=qdl_code,
                                 ticker=ticker,
                                 qdl_exchange=exchange))

            else:
                print(f'Warning: Could not extract exchange and ticker for \'{qdl_code}\'.')

    return meta


class IBapi(EWrapper, EClient):

    def __init__(self):
        EClient.__init__(self, self)
        self.requests = dict()

    def symbolSamples(self, reqId: int, contractDescriptions):

        super().symbolSamples(reqId, contractDescriptions)

        Candidiate = namedtuple('Candidiate', 'symbol type primary_exchange currency')
        candidiates = list()

        symbol = self.requests[reqId]
        for desc in contractDescriptions:
            deriv_sec_types = ''
            candidiates.append(Candidiate(desc.contract.symbol,
                                          desc.contract.secType,
                                          desc.contract.primaryExchange,
                                          desc.contract.currency))

        candidiates = list(filter(lambda c: c.symbol == symbol and c.currency == 'USD', candidiates))

        for c in candidiates:
            if c.primary_exchange not in EXCHANGES:
                print(f'{symbol} -> {c.primary_exchange}')
                exit()

        candidiates = list(filter(lambda c: c.primary_exchange in EXCHANGES_KEEP, candidiates))

        if len(candidiates) > 1:
            print(f'{symbol} more than 1 candidiate')
            pprint(candidiates)
            print()
            exit()

        pprint(candidiates)


def prepare_database():
    db = sqlite3.connect(SYMBOLS_DB_PATH)

    db.execute('PRAGMA foreign_keys=ON;')

    db.execute('''
CREATE TABLE IF NOT EXISTS symbols_meta(
    qdl_code CHAR(32) PRIMARY KEY,
    qdl_ticker CHAR(32) NOT NULL,
    qdl_name CHAR(256) NOT NULL,
    qdl_exchange CHAR(16) NOT NULL,
    qdl_type CHAR(8) NOT NULL,
    last_trade DATE,
    start_date DATE,
    end_date DATE,
    ib_symbol CHAR(32),
    ib_type CHAR(8),
    ib_primary_exchange CHAR(16),
    ib_currency CHAR(8),
    is_common_stk BOOL,
    is_tradable BOOL
    );''')

    db.close()


def main():

    prepare_database()
    meta = get_quandl_tickers()

    print(meta[0:2])
    exit()

#    pprint(meta)
    app = IBapi()
    app.connect('127.0.0.1', 4002, 1)

    api_thread = threading.Thread(target=lambda: app.run(), daemon=True)
    api_thread.start()

    sleep(1)

    for idx, meta_row in enumerate(meta):
        symbol = meta_row.ticker.replace('_', ' ')
#        symbol = meta_row.ticker
#        symbol = 'BRK'

        print(f'{idx+1} of {len(meta)} -> {symbol}')

        app.requests[idx] = symbol
        app.reqMatchingSymbols(idx, symbol)
        sleep(1.1)

    app.disconnect()


if __name__ == '__main__':
    main()
