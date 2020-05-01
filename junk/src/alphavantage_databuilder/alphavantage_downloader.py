import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

from db_connectors import SQLite3Connector
from alphavantage import AlphaVantageApi
from pathlib import Path
from time import sleep
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import datetime


class AlphaVantageDownloader:
    """
    This class downloads stock data from Alpha Vantage and stores it in a sqlite3 database.

    Internally `AlphaVantageApi` is used to request data through Alpha Vantage's API. Data
    is saved to a sqlite3 database using `SQLite3Connector`.

    """

    _meta_table = None
    _daily_adjusted_table = None

    _pool = ThreadPoolExecutor(1)
    _db_lock = Lock()
    _requests_lock = Lock()
    _requests = list()

    _is_processing_flag = False
    _lock = Lock()
    _process_task_fut = None

    def __init__(self,
                 api_key: str,
                 data_path: Path,
                 db_file: Path,
                 meta_table: str = 'symbols_meta',
                 daily_adjusted_table: str = 'daily_adjusted'):
        """
        Constructs an instance of `AlphaVantageDownloader`.

        Args:
            api_key:                Alpha Vantage API key to use.
            data_path:              Path to data directory.
            db_file:                Relative path of sqlite3 database file.
            meta_table:             Name of meta table.
            daily_adjusted_table:   Name of daily adjusted table.

        """

        self._meta_table = meta_table
        self._daily_adjusted_table = daily_adjusted_table

        # Check that data directory exists.
        if not data_path.is_dir():
            raise NotADirectoryError(
                f'data_path: \'{data_path}\' is not valid directory.')

        # Connect to database; create if it doesn't exist.
        self._db_path = data_path / db_file
        db = SQLite3Connector.connect(self._db_path)

        # Create meta table if it doesn't exist.
        meta_cols = [
            {'name': 'id', 'dtype': 'INTEGER', 'not_null': True, 'pk': True},
            {'name': 'ts_type', 'dtype': 'CHAR', 'size': 32, 'not_null': True},
            {'name': 'description', 'dtype': 'CHAR', 'size': 128, 'not_null': True},
            {'name': 'symbol', 'dtype': 'CHAR', 'size': 16, 'not_null': True},
            {'name': 'last_refreshed', 'dtype': 'DATE', 'not_null': True},
            {'name': 'outputsize', 'dtype': 'CHAR', 'size': 8, 'not_null': True},
            {'name': 'timezone', 'dtype': 'CHAR', 'size': 32, 'not_null': True},
        ]
        db.create_table(meta_table, meta_cols)

        # Create daily adjusted table if it doesn't exist.
        daily_adj_cols = [
            {'name': 'id', 'dtype': 'INTEGER', 'not_null': True, 'pk': True},
            {'name': 'symbol', 'dtype': 'CHAR', 'size': 16, 'not_null': True},
            {'name': 'date', 'dtype': 'DATE'},
            {'name': 'open', 'dtype': 'DECIMAL', 'size': 2},
            {'name': 'high', 'dtype': 'DECIMAL', 'size': 2},
            {'name': 'low', 'dtype': 'DECIMAL', 'size': 2},
            {'name': 'close', 'dtype': 'DECIMAL', 'size': 2},
            {'name': 'adj_close', 'dtype': 'DECIMAL', 'size': 2},
            {'name': 'volume', 'dtype': 'DECIMAL', 'size': 1},
            {'name': 'divident', 'dtype': 'DECIMAL', 'size': 1},
            {'name': 'split', 'dtype': 'FLOAT'},
        ]
        db.create_table(daily_adjusted_table, daily_adj_cols)

        db.close()

        # Initialize Alpha Vantage API connector.
        av_api_state_path = data_path / 'av_api_state.json'
        self._av_api = AlphaVantageApi(api_key=api_key,
                                       max_calls_per_min=5,
                                       max_calls_per_day=500,
                                       state_json_path=av_api_state_path,
                                       load_state_enabled=True,
                                       save_state_enabled=True)

    def download_daily_adj(self,
                           symbol: str,
                           overwite: bool = False):
        """
        Downloads the daily adjusted stock prices for some symbol.

        Args:
            symbol:     Name of symbol to query.
            overwite:   Enable overwriting of existing records in database.

        """

        # If overwite is not enabled, check if record exists.
        if not overwite:

            # Search for daily adjusted record matching symbol.
            cols = None
            filter_clause = f'WHERE symbol == "{symbol}" and ts_type == "daily_adj"'
            with self._db_lock:
                db = SQLite3Connector.connect(self._db_path)
                res = db.select(self._meta_table,
                                w_filter=filter_clause)
                db.close()

            # Output info and return to caller; query is skipped.
            if len(res) > 0:
                print(f'INFO: Record daily-adjusted for \'{symbol}\' exists.')
                print('      Overwrite off, skipping.')

                return

        # Otherwise, if overwrite is enabled, delete exisiting matching records.
        else:

            with self._db_lock:

                # Delete existing matching records.
                db = SQLite3Connector.connect(self._db_path)
                filter_clause = f'WHERE symbol == "{symbol}" and ts_type == "daily_adj"'
                db.delete(self._meta_table, filter_clause)
                filter_clause = f'WHERE symbol == "{symbol}"'
                db.delete(self._daily_adjusted_table, filter_clause)
                db.commit()
                db.close()

        # Perform query for symbol.
        print(f'INFO: Requesting daily-adjusted for \'{symbol}\'...')

        # Lock to prevent concurrent modification to requests list.
        with self._lock:

            # Query future is enqued to executor and added to requests.
            self._requests.append({'ts_type': 'daily_adj',
                                   'symbol': symbol,
                                   'request': self._av_api.get_daily_adj(symbol),
                                   'processed': False
                                   })

            # Start processing task if not currently running.
            if not self._is_processing_flag:
                self._process_task_fut = self._pool.submit(self._process)

    def propogate_exceptions(self):
        """
        In order for exceptions to propogate from other thread, it is required to call 
        `result()` on the process task future.
        """

        if self._process_task_fut is not None:
            _ = self._process_task_fut.result()

    def _process(self):
        """
        Process task which monitors status of active request futures.
        """

        # Set is processing flag.
        with self._lock:
            self._is_processing_flag = True

        # This counter is needed to limit logging output interval.
        log_i = 0

        # Loop while not all requests are processed.
        while any(not t['processed'] for t in self._requests):

            # todo: check for duplicate in flight requests.

            # Look for completed requests and save them to the database.
            for t in self._requests:
                if t['request'].done() and not t['processed']:
                    print(
                        f'INFO: Processing response for \'{t["ts_type"]}\'', end='')
                    print(f' for \'{t["symbol"]}\'.')
                    t['processed'] = True
                    self._save_to_db(t)

            # Output completion status info.
            if log_i % 10 == 0:
                done_count = sum(t['request'].done() for t in self._requests)
                print('INFO: {} / {} : {:.0f}% requests remaining.'.format(
                    done_count,
                    len(self._requests),
                    100. * done_count / len(self._requests)))

                # If Alpha Vantage API has exceeded call frequency limits, printing remaining
                # time on timer.
                quota_info = self._av_api.quotas()
                if quota_info['wait']:
                    t = datetime.timedelta(
                        seconds=round(quota_info['timeout']))
                    print(f'      Waiting {t}')

            log_i += 1

            sleep(1)

        print(f'INFO: Finished processing {len(self._requests)} requests.')

        # When all requests are processed, clear requests list, and reset flag.
        with self._lock:
            if len(self._requests) == 0:
                self._requests.clear()
                self._is_processing_flag = False

    def _save_to_db(self, task):
        """
        Saves a completed request to the sqlite3 database.
        """

        ts_type = task['ts_type']
        symbol = task['symbol']
        request = task['request']
        processed = task['processed']

        assert(symbol != '')
        assert(request.done())
        assert(processed)

        # Get result of request future.
        result = request.result()

        print(f'INFO: Writing \'{ts_type}\' for \'{symbol}\' to datebase.')

        # Case 1: Request type if for daily adjusted data.
        if ts_type == 'daily_adj':

            # Unpack data and meta data from request result.
            data, meta = result

            # Extract description, last refresh, output size, and timezone.
            meta_description = meta['1. Information']
            assert(meta['2. Symbol'] == symbol)
            meta_last_refresh = meta['3. Last Refreshed']
            meta_ouputsize = meta['4. Output Size']
            meta_timezone = meta['5. Time Zone']

            if symbol == 'GS':
                print(f'**GS = {len(data)}')

            # Lock sqlite3 database to prevent concurrent access.
            with self._db_lock:

                # Insert data into symbol metadata table.
                cols = ('id', 'ts_type', 'description', 'symbol', 'last_refreshed',
                        'outputsize', 'timezone')
                values = [(None, ts_type, meta_description, symbol, meta_last_refresh,
                           meta_ouputsize, meta_timezone)]
                db = SQLite3Connector.connect(self._db_path)
                db.insert(self._meta_table,
                          cols, values)

                # Form data for insertion into daily adjusted table.
                vals = list([(None,
                              symbol,
                              data.index[i].strftime('%Y-%m-%d'),
                              data.values[i][0],
                              data.values[i][1],
                              data.values[i][2],
                              data.values[i][3],
                              data.values[i][4],
                              data.values[i][5],
                              data.values[i][6],
                              data.values[i][7]
                              ) for i in range(len(data))
                             ])

                cols = ('id', 'symbol', 'date', 'open', 'high', 'low', 'close',
                        'adj_close', 'volume', 'divident', 'split')

                # Insert daily adjusted stock prices into table.
                db.insert(self._daily_adjusted_table,
                          cols, vals)

                db.commit()
                db.close()

        # Case Z: Other data types are not yet supported.
        else:
            assert(False)

    def done(self):
        """
        Must be called to cleanup resources and save `AlphaVantageApi` state.
        """

        self._av_api.done()


# api_key = '63D5Z3C3GBRHSENV'
# api_key = '5SSRQTB3PVHK28H9'
api_key = 'AG0GIQJL6LA5U2Q6'
data_path = Path(__file__).absolute().parent.parent.parent / 'data'
print(data_path)
downloader = AlphaVantageDownloader(api_key=api_key,
                                    data_path=data_path,
                                    db_file='av.db')

syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT',
        'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
        'XOM', 'GS', 'HD',
        'IBM', 'INTC', 'JNJ', 'MCD', 'MRK',
        'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
        'UTX', 'UHN', 'VZ', 'V', 'WBA', 'WMT']

for s in syms:
    downloader.download_daily_adj(s, overwite=False)

downloader.propogate_exceptions()

downloader.done()
