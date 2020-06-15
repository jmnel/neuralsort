from pathlib import Path

ROOT_DIRECTORY = Path(__file__).absolute().parents[2]
DATA_DIRECTORY = ROOT_DIRECTORY / 'data'
DATABASE_NAME = 'iex_tops.sqlite3'
