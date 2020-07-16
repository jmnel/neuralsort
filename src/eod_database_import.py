import settings
from quandl_import import *
from vix_import import import_vix
from sp500_import import import_sp500

logger = logging.getLogger(__name__)

QUANDL_DATABASE_PATH = settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME


def prepare_database(db: sqlite3.Connection):

    logger.info('Preparing qdl.sqlite3 database.')
    db.execute('PRAGMA foreign_keys = ON;')

    logger.info('Deleting qdl_eod_symbol_index index.')
    db.execute('''
DROP INDEX IF EXISTS qdl_eod_symbol_index;
''')

    logger.info('Deleting qdl_eod table.')
    db.execute('''
DROP TABLE IF EXISTS qdl_eod;
''')

    logger.info('Deleting qdl_symbols table.')
    db.execute('''
DROP TABLE IF EXISTS qdl_symbols;
''')

    logger.info('Creating qdl_symbol table.')
    # Create meta table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_symbols(
    symbol CHAR(32) PRIMARY KEY,
    qdl_code CHAR(32) NOT NULL,
    name CHAR(256) NOT NULL,
    exchange CHAR(16) NOT NULL,
    last_trade DATE,
    start_date DATE,
    end_date DATE,
    lifetime_returns FLOAT,
    is_common BOOLEAN
    );
''')

    logger.info('Creating qdl_eod table.')
    # Create data table.
    db.execute('''
CREATE TABLE IF NOT EXISTS qdl_eod(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL,
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    dividend FLOAT NOT NULL,
    split FLOAT NOT NULL,
    adj_open FLOAT NOT NULL,
    adj_high FLOAT NOT NULL,
    adj_low FLOAT NOT NULL,
    adj_close FLOAT NOT NULL,
    adj_volume FLOAT NOT NULL,
    FOREIGN KEY(symbol) REFERENCES qdl_symbols(symbol)
);''')

    # Create symbol EOD index.
    logger.info('Creating qdl_eod_symbol_index index.')
    db.execute('''
CREATE INDEX qdl_eod_symbol_index ON qdl_eod(symbol);
''')


def main():

    with sqlite3.connect(QUANDL_DATABASE_PATH) as db:
        prepare_database(db)
        get_quandl_tickers(db)
        bulk_download(db)
        generate_meta_data(db)
        purge_empty(db)
    import_vix()
    import_sp500()
    logger.info('Done.')


if __name__ == '__main__':
    main()
