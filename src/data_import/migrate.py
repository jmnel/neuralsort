import sqlite3
import json
from pathlib import Path
from datetime import datetime, date

import settings

QUANDL_DATABASE_PATH = settings.DATA_DIRECTORY / settings.QUANDL_DATABASE_NAME
INFO_PATH = settings.DATA_DIRECTORY / settings.QUANDL_IMPORT_INFO_FILE


def should_migrate() -> bool:
    """
    Determines if the database needs to be migrated.

    Migration occurs if any of the following are true:
        1. json decode fails
        2. file missing
        3. dictionary keys mismatch
        4. no entries

    Returns:
        bool:   Indicates if migration should occur.

    """

    # Check if there is database to migrate.
    if not QUANDL_DATABASE_PATH.is_file():
        print(f'No existing database {QUANDL_DATABASE_PATH} to migrate.')

        # Delete info json if it exists; something went wrong with previous migration.
        if INFO_PATH.is_file():
            INFO_PATH.unlink()
        return False

    # Check for existing info json file.
    if INFO_PATH.is_file():

        # Try to open and decode the json.
        try:
            with open(INFO_PATH) as conf_file:
                info = json.loads(conf_file.read())

        except JSONDecodeError as e:
            print(f'{INFO_PATH} is  corrupted.')
            INFO_PATH.unlink()
            return True

        # Decoding json succeeded.
        else:

            # Check that entries have correct keys.
            for entry in info:
                EXPECTED_KEYS = {'date',
                                 'last_refresh_date',
                                 'size',
                                 'num_symbols',
                                 'num_days',
                                 'version',
                                 'type'}
                if set(entry.keys()) != EXPECTED_KEYS:
                    print(f'{INFO_PATH} is corrupted. Bad keys.')
                    INFO_PATH.unlink()
                    return True

            # Check for existing entries.
            if len(info) > 0:
                print(f'Already migrated. {INFO_PATH} has {len(info)} entries.')
                return False

    return True


def main():

    print('Migrating to new EOD database update mechanism.')

    # Check if migration is needed.
    if not should_migrate():
        return

    # Get file modify timestamp.
    mtime = datetime.fromtimestamp(QUANDL_DATABASE_PATH.lstat().st_mtime).timestamp()

    # Get  last date in database.
    print('Finding last update from database.')
    db = sqlite3.connect(QUANDL_DATABASE_PATH)

    last_date = db.execute('SELECT MAX(date) FROM qdl_eod;').fetchall()[0][0]
    num_days = db.execute('SELECT COUNT(DISTINCT date) FROM qdl_eod;').fetchall()[0][0]

    print(f'Last refresh occured on {last_date}.')
    version = settings.QUANDL_DATABASE_VERSION
    print(f'Assuming database version {version}.')

    # Count symbols in database.
    num_symbols = db.execute('SELECT COUNT(symbol) FROM qdl_symbols;').fetchall()[0][0]

    db.close()

    # Get download size from old download file if it exists.
    old_download = settings.DATA_DIRECTORY / 'quandl' / 'EOD.zip'
    if old_download.is_file():
        size = old_download.lstat().st_size
    else:
        size = -1

    # Create info json file with existing database.
    info = [{'date': last_date,
             'last_refresh_date': mtime,
             'size': size,
             'num_symbols': num_symbols,
             'num_days': num_days,
             'version': version,
             'type': 'migrate'}]

    # Dump info to json.
    print(f'Saving info to {INFO_PATH}.')
    with open(INFO_PATH, 'w') as conf_file:
        conf_file.write(json.dumps(info, indent=4, sort_keys=True))

    print('Migration completed.')


if __name__ == '__main__':
    main()
