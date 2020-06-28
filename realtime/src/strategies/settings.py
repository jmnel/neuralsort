from pathlib import Path
import os
import logging

ROOT_DIRECTORY = Path(__file__).absolute().parents[2]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'
IEX_DATABASE_NAME = 'iex_tops.sqlite3'
QDL_DATABASE_NAME = 'neuralsort.db'

QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
ALPHA_VANTAGE_API_KEY = '63D5Z3C3GBRHSENV'

EXCHANGES = {'NASDAQ', 'NYSE', 'NYSE', 'NYSE MKT', 'NYSE Arca'}

LOG_LEVEL = logging.INFO


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


setup_logger()
