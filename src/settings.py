import os
from pathlib import Path
import re
import logging

ROOT_DIRECTORY = Path(__file__).absolute().parents[1]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'

QUANDL_DATABASE_NAME = 'qdl.sqlite3'
QUANDL_DATABASE_VERSION = '0.1'
QUANDL_IMPORT_INFO_FILE = 'qdl_import_info.json'

IMPORT_QUANDL_RAW = True


QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
ALPHA_VANTAGE_API_KEY = '63D5Z3C3GBRHSENV'

#NASDAQ_TEST_TICKERS = ['ZVZZT']


EXCHANGES = {'NASDAQ', 'NYSE', 'NYSE', 'NYSE MKT', 'NYSE Arca'}


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


setup_logger()
