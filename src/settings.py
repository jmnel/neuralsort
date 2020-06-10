import os
from pathlib import Path
import re

ROOT_DIRECTORY = Path(__file__).absolute().parents[1]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'

DATABASE_NAME = 'neuralsort.db'
EODDATA_DATABASE_NAME = 'eoddata.db'

IB_DATABASE_NAME = 'ib_bbo.db'

IMPORT_QUANDL_RAW = True


QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
ALPHA_VANTAGE_API_KEY = '63D5Z3C3GBRHSENV'

#NASDAQ_TEST_TICKERS = ['ZVZZT']


EXCHANGES = {'NASDAQ', 'NYSE', 'NYSE', 'NYSE MKT', 'NYSE Arca'}
