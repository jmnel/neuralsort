from pathlib import Path
import os
import logging

ROOT_DIRECTORY = Path(__file__).absolute().parents[2]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'
DATABASE_NAME = 'iex_tops.sqlite3'

QUANDL_API_KEY = os.getenv('QUANDL_API_KEY')
ALPHA_VANTAGE_API_KEY = '63D5Z3C3GBRHSENV'

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
