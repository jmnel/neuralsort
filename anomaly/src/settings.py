from pathlib import Path
import re

ROOT_DIRECTORY = Path(__file__).absolute().parents[1]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'

EXCHANGE_NAMES = {'AMEX', 'NASDAQ', 'NYSE', 'OTCBB'}

DATABASE_NAME = 'neuralsort.db'

IMPORT_EOD_RAW = True

EOD_NON_COMMON_PATTERN = re.compile('^[A-Z]+\.')
