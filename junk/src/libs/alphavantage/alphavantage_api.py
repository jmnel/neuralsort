from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep, time
import json
from pathlib import Path


class AlphaVantageApi:
    """
    API wrapper to allow asyncronous downloading of Alpha Vantage realtime and historical
    data.

    """

    _pool = ThreadPoolExecutor(6)
    _max_calls_per_min = 0
    _max_calls_per_day = 0
    _api_key = None

    _min_wait = 60 + 10
#    _min_wait = 20
    _day_wait = 60 * 60 * 24 + 60 * 5

    _time_prev_min = 0
    _time_prev_day = 0

    _lock = Lock()
    _wait_flag = False

    _process_task = None

    _quota_min = 0
    _quota_day = 0

    _av_api = None

    _state_json_path = None

    _load_state_enabled = False
    _save_state_enabled = False

    _num_retries = 3
    _short_retry_timeout = 120
    _long_retry_timeout = 0

    _api_output_format = None

    def __init__(self,
                 api_key: str,
                 max_calls_per_min: int = 5,
                 max_calls_per_day: int = 500,
                 state_json_path: Path = None,
                 #                 state_json_path: str = None,
                 load_state_enabled: bool = False,
                 save_state_enabled: bool = False,
                 output_format: str = 'pandas'):
        """
        Constructs Alpha Vantage API wrapper.

        Args:
            api_key:                API key
            max_calls_per_min:      limit of number of API calls/minute
            max_calls_per_day:      limit of number of API calls/day
            state_json_path:         location to store API call counts and timers
            load_state_enabled:      turns on info json loading
            save_state_enabled:      turns on info json saving
            output_format:          output format; either 'pandas' or 'csv'

        """

        self._api_key = api_key
        self._max_calls_per_min = max_calls_per_min
        self._max_calls_per_day = max_calls_per_day

        # Initialize minute and day timers with current time.
        self._time_prev_min = time()
        self._time_prev_day = time()

        # Check that output format is either 'pandas' or 'csv'.
        if output_format not in ('pandas', 'csv'):
            raise ValueError(
                'output_format must be either \'pandas\' or \'csv\'.')

        # Create Alpha Vantage time series object.
        self._av_api = TimeSeries(api_key, output_format=output_format)

        self._load_state_enabled = load_state_enabled
        self._save_state_enabled = save_state_enabled
        self._state_json_path = state_json_path

        self._api_output_format = output_format

        # Load json configuration if enabled.
        if load_state_enabled:
            self._load_info()

    def done(self):
        """
        Done must be called to cleanup resources and save state to json file.
        """

        # Save json configuration if enabled.
        if self._save_state_enabled:
            self._save_info()

    # Loads information from json file.
    def _load_info(self):

        # Throw exception if json file path was not provided.
        if self._state_json_path is None:
            raise ValueError(
                'JSON info file path must be provided with load_info enabled.')

        # Check that json file has valid parent directory.
        if not self._state_json_path.parent.is_dir():
            error_msg = 'JSON state file: {} must have valid parent directory.'
            error_msg = error_msg.format(self._state_json_path)
            raise ValueError(error_msg)

        # Try to open information json file and decode.
        try:
            with open(self._state_json_path, 'r') as json_file:
                json_info = json.load(json_file)

        # Catch non-fatal missing file and json decoder exceptions.
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(f'JSON file {self._state_json_path} not found or corrupted.')
            print('Not loading JSON info.')

        else:

            # Check that json state file has correct format.
            if ('api_key' not in json_info or
                    'quota_min' not in json_info or
                    'quota_day' not in json_info or
                    'time_prev_min' not in json_info or
                    'time_prev_day' not in json_info):

                print(
                    f'JSON file {self._state_json_path} corrupted.')
                print('Not loading JSON info.')
                return

            # Check if API key has changed.
            if self._api_key == json_info['api_key']:

                # Store information values from json file to this class.
                self._quota_min = json_info['quota_min']
                self._quota_day = json_info['quota_day']
                self._time_prev_min = json_info['time_prev_min']
                self._time_prev_day = json_info['time_prev_day']

            # Don't load state from json file if API key has changed.
            else:
                print('Warning: API key changed, not loading state:')
                print(f'\tAPI key is \'{self._api_key}\'')
                print(f'\tPreviosly API key was \'{json_info["api_key"]}\'')

    # Saves the current state information to json file.
    def _save_info(self):

        # Ensure that json file path was provided if save is enabled.
        if self._state_json_path is None:
            raise ValueError(
                'JSON info file path must be provided with save_info enabled.')

        # Check that json file has valid parent directory.
        if not self._state_json_path.parent.is_dir():
            raise ValueError(
                'JSON state file must have valid parent directory.')

        # Build dictionary of usage quotas and timers.
        info_data = {'api_key': self._api_key,
                     'quota_min': self._quota_min,
                     'quota_day': self._quota_day,
                     'time_prev_min': self._time_prev_min,
                     'time_prev_day': self._time_prev_day}

        # Write information to file.
        with open(self._state_json_path, 'w') as json_file:
            json.dump(info_data, json_file)

    # Updates quotas and their associated timers.
    def _update_quotas(self):

        # Get elapsed time for minute and day timers.
        time_delta_min = time() - self._time_prev_min
        time_delta_day = time() - self._time_prev_day

        # If minute timer has expires, reset minute quota.
        if time_delta_min > self._min_wait:
            with self._lock:
                self._time_prev_min = time()
                self._quota_min = 0

        # If day timer has expired, reset day quota.
        if time_delta_day > self._day_wait:
            with self._lock:
                self._time_prev_day = time()
                self._quota_day = 0

        # Set wait flag.
        with self._lock:
            self._wait_flag = not(self._quota_min < self._max_calls_per_min
                                  and self._quota_day < self._max_calls_per_day)

    # Runs proivded function if quota has not been exceeded; otherwise waits.
    def _task(self, task_fn, **kargs):

        # Keep waiting until quota is available.
        while True:

            # Poll quotas.
            self._update_quotas()

            if not self._wait_flag:

                # Increment minute and day quotas.
                with self._lock:
                    self._quota_min += 1
                    self._quota_day += 1

                # Run task.
                return task_fn(**kargs)

        sleep(0.1)

    # Calls 'get_daily_adjusted' with Alpha Vantage API.
    def _daily_adj_api(self, **kargs):

        for attempt in range(self._num_retries):
            try:
                response = self._av_api.get_daily_adjusted(**kargs)

            except ValueError as e:
                print(
                    f'WARNING: Alpha Vantage API call frequency exceeded attempt {attempt}.')
                print(
                    f'         Trying again in {self._short_retry_timeout} seconds.')
                success = False
                sleep(self._short_retry_timeout)

            else:
                success = True
                return response

        raise ValueError('ERROR: Alpha Vantage API call frequency exceeded.')

        #        sleep(2)
        #        return ('adsf', [1, 2, 3])
#        return self._av_api.get_daily_adjusted(**kargs)

    def get_daily_adj(self,
                      symbol: str,
                      outputsize='full'):
        """
        Concurrently get daily adjusted prices for symbol and returns future.

        This function waits until call quota is available to respect Alpha Vantage's API's
        per-minute and per-day call limits. A future is spawned and returned, which willl
        eventually hold the result of the query.

        Internally, get_daily_adjusted from alpha_vantage API is called. The daily adjusted
        prices are corrected for splits and dividents.

        Args:
            symbol:         symbol of security to query
            outputsize:     size of data to return; either 'full' or 'compactsize'

        Returns:
            future that will eventually hold result of query

        """

        # Check that symbol is non-empty string.
        if symbol in ('', None) and isinstance(symbol, str):
            raise ValueError('symbol must be non-empty string.')

        # Check that outputsize is either 'full' or 'compact'.
        if outputsize not in ('full', 'compact'):
            raise ValueError(
                'outputsize must be either \'full\' or \'compact\'.')

        # Submit task in executor pool.
        t = self._pool.submit(self._task,
                              self._daily_adj_api,
                              symbol=symbol,
                              outputsize=outputsize)

        return t

    def quotas(self):
        """
        Get state information regarding quotas which implement policy to respect per-day and
        per-minute API call limits.

        Returns:
            dict containing quota information and timers

        """

        # Store current state in dictionary.
        info = {'per_min_remaining': self._max_calls_per_min - self._quota_min,
                'per_day_remaining': self._max_calls_per_day - self._quota_day,
                'count_min': self._quota_min,
                'count_day': self._quota_day,
                'max_calls_per_min': self._max_calls_per_min,
                'max_calls_per_day': self._max_calls_per_day}

        # If the wait flag is set, also store timeout.
        if self._wait_flag:

            # Calcluate per-minute and per-day timeouts.
            timeout_min = self._time_prev_min - time() + self._min_wait
            timeout_day = self._time_prev_day - time() + self._day_wait

            info['wait'] = True

            # The daily quota has been exceeded, store time until daily timer resets.
            if self._quota_day >= self._max_calls_per_day:
                info['timeout'] = timeout_day

            # Otherwise, per-minute quota has been exceed, store time until minute
            # timer resets.
            elif self._quota_min >= self._max_calls_per_min:
                info['timeout'] = timeout_min

            # Sanity check; delete this eventually.
            else:
                # Should not fall through to this.
                assert(False)

        # Wait flag is not set.
        else:

            info['wait'] = False

        return info


# av = AlphaVantageApi(api_key='5SSRQTB3PVHK28H9',
#                     load_state_enabled=True,
#                     save_state_enabled=True,
#                     state_json_path='alpha_vantage_info.json')

#syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'KO']
# syms = ['MMM', 'AXP']

#tasks = [(s, av.get_daily_adj(s, outputsize='compact')) for s in syms]

#done = False
# while not done:

#    info = av.quotas()

#    if info['wait']:
#        print(info)
#        print('Waiting for {:.2f} seconds...'.format(info['timeout']))

#    print(sum(not t[1].done() for t in tasks))

#    done = all(t[1].done() for t in tasks)

#    sleep(2)

# av.done()

# tasks = [av.get_daily_adj(str(i)) for i in range(50)]

# syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'KO']
# syms = ['MMM', 'AXP']

# tasks = [(s, av.get_daily_adj(s, outputsize='compact')) for s in syms]

# while any(not t[1].done() for t in tasks):
#    sleep(5)
#    for i, t in enumerate(tasks):
#        if t[1].done():
#            print(f'{t[0]} is done')
#            tasks.pop(i)
#    if av._wait_flag:
#        print('Waiting for {:.0f} seconds.'.format(
#            65 + av._time_prev_min - time()))
#    sleep(2)


# av.done()
#    print(sum(not t[1].done() for t in tasks))
#    sleep(2)

# data = [t.result()[0] for t in tasks]
# meta = [t.result()[1] for t in tasks]

# for m in meta:
#    print(m)


# syms = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT',
#        'CVX', 'CSCO', 'KO', 'DIS', 'DOW',
#        'XOM', 'GS', 'HD', 'GS', 'HD',
#        'IBM', 'INTC', 'JNJ', 'MCD', 'MRK',
#        'MSFT', 'NKE', 'PFE', 'PG', 'TRV',
#        'UTX', 'UHN', 'VZ', 'V', 'WBA', 'WMT']
