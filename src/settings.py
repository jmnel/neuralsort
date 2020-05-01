from pathlib import Path
import re

ROOT_DIRECTORY = Path(__file__).absolute().parents[1]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'

DATABASE_NAME = 'neuralsort.db'
EODDATA_DATABASE_NAME = 'eoddata.db'

IMPORT_QUANDL_RAW = True


QUANDL_API_KEY = 'yRyzMHs_wg6bMPAcExUS'

#NASDAQ_TEST_TICKERS = ['ZVZZT']


EXCHANGES = {'NASDAQ', 'NYSE', 'NYSE', 'NYSE MKT', 'NYSE Arca'}
