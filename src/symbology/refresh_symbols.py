import sqlite3
import requests
import io
import zipfile
import csv
from pprint import pprint
import re

import ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.client import MarketDataTypeEnum
from ibapi.ticktype import TickTypeEnum

import settings

QUANDL_META_ENDPOINT = 'https://www.quandl.com/api/v3/databases/EOD/metadata?api_key={}'

response = requests.get(QUANDL_META_ENDPOINT.format(settings.QUANDL_API_KEY))
if not response.ok:
    print(f'Quandl metadata endpoint request error {response.status_code}')
    exit()

ss = io.BytesIO(response.content)
zf = zipfile.ZipFile(ss)

with zf.open('EOD_metadata.csv', 'r') as f:
    f2 = io.TextIOWrapper(f, encoding='utf-8')
    reader = csv.reader(f2, delimiter=',')
    header = next(reader)

    exchange_regex = re.compile(r'<p><b>Exchange</b>: (.*?)</p>')
    ticker_regex = re.compile(r'<p><b>Ticker</b>: (.*?)</p>')

    invalid = 0
    for row in reader:
        symbol = row[0]

        if symbol[-1] == '_':
            continue

        desc = row[1][:-35 - len(symbol) - 3]
        m1 = exchange_regex.search(row[2])
        m2 = ticker_regex.search(row[2])

        s.add(symbol)

        if symbol[-1] == '_':
            trailing.add(symbol)

        if m1 and m2:
            exchange = m1.groups()[0]
            ticker = m2.groups()[0]