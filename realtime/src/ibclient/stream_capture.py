import sqlite3
from pprint import pprint
from datetime import datetime, time
import signal

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.client import MarketDataTypeEnum
from ibapi.ticktype import TickTypeEnum

import threading
from time import sleep, time_ns
from queue import Queue

import settings


class IBapi(EWrapper, EClient):
    """
    Client that connects to IB Gateway and handles TWS API queries.

    """

    def __init__(self, msg_queue: Queue):
        EClient.__init__(self, self)
        self.messages = msg_queue

    def tickPrice(self, reqId, tickType, price, attrib):
        """
        Handle a price tick message.

        """

        # Filter for DELAYED_LAST_PRICE (68) ticks.
        if tickType == 68:

            # Place tick in message queue.
            self.messages.put((reqId,
                               0,
                               price))

    def tickSize(self, reqId, tickType, size):
        """
        Handle a size tick message.

        """

        # Filter for DELAYED_LAST_SIZE (71) ticks.
        if tickType == 71:

            # Place tick in message queue.
            self.messages.put((reqId,
                               1,
                               size))

    def tickString(self, reqId, tickType, value):
        if tickType == 88:
            self.messages.put((reqId, 2, int(value)))


class StreamListener:
    """
    Uses a client to subscribe, capture, and store trade report ticks from the TWS API.

    """

    def __init__(self):
        self.msg_queue = Queue()
        self.is_running = False

        self.today = datetime.now().strftime('%Y-%m-%d')

        # Set shutdown time to 5PM.
        self.shutdown_time = time(hour=17, minute=0, second=0)

    def start(self):
        """
        Start capturing the market streaming data.

        """

        print('Starting capture')

        self.is_running = True

        # Initialize client and connect to IB Gateway socket.
        self.app = IBapi(self.msg_queue)
        self.app.connect('127.0.0.1', 4002, 1)

        # Launch client thread.
        api_thread = threading.Thread(target=self.app.run, daemon=True)
        api_thread.start()

        # Give client thread time to establish connection and catch up.
        sleep(1)

        # Set data type to delayed.
        self.app.reqMarketDataType

        # Subscribe to market data streams using watchlist.
        self.init_watch()

        self.db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME)

        # Run appliction main loop.
        while(self.is_running):

            # Automatically shutdown after market close.
            if datetime.now().time() > self.shutdown_time:
                print('Market now closed')
                self.is_running = False
                assert self.msg_queue.empty()
                break

            # Process tick queue.
            while not self.msg_queue.empty():

                # Pop tick from queue.
                tick = self.msg_queue.get()

                request_id = tick[0]
                symbol = self.watch_list[request_id][1].symbol

                # Trade reports are generated when a size tick is received. We use a particular request ticker's
                # last price.

                # Tick is price type.
                if tick[1] == 0:
                    price = tick[2]

                    # Update price ticker.
                    self.last_price[request_id] = price

                if tick[1] == 2:
                    ts = tick[2]
                    self.last_time[request_id] = ts

                # Tick is size type.
                elif tick[1] == 1:
                    size = tick[2]

                    # A last price tick should always proceed a size tick.
                    assert self.last_price[request_id] is not None
                    assert self.last_time[request_id] is not None

                    print(f'{symbol}: ${price}, {size} shares')

                    # Store trade report in database.
                    t = self.last_time[request_id]
                    row = (self.today, t, symbol, int(100. * self.last_price[request_id]), size)
                    self.db.execute('''
INSERT INTO ib_trade_reports(day, timestamp, symbol, price, size)
VALUES(?, ?, ?, ?, ?);''', row)
                    self.db.commit()

            # Wait a bit for the queue to fill up again.
            sleep(5)

        print('Stopping capture')

        # Unsubscribe, disconnect client, close database connection.
        self.stop_watch()
        self.app.disconnect()
        self.db.close()

    def stop(self):
        """
        Gracefully stop capturing messages; exits main listening loop.

        """

        self.is_running = False

    def init_watch(self):
        """
        Initialize market data streams using watchlist database.

        """

        # Use today's date.
        date = datetime.now().strftime('%Y-%m-%d')

        # Get watchlist and associated Smart Routing information from database.
        db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME)
        rows = db.execute('''
    SELECT symbol, type, primary_exchange, currency FROM ib_watch WHERE date=? ORDER BY symbol;
    ''', (date,)).fetchall()
        db.close()

        self.watch_list = list()

        # Set market data type to delayed since this is a paper trading session.
        self.app.reqMarketDataType(MarketDataTypeEnum.DELAYED)
        sleep(5)

        # Initialize last price ticker list with None.
        self.last_price = list(None for _ in range(len(rows)))
        self.last_time = list(0 for _ in range(len(rows)))

        # Loop through each symbol in watchlist.
        for request_id, r in enumerate(rows):

            # Create contract.
            contract = Contract()
            contract.symbol = r[0]
            contract.secType = r[1]

            # Use the IB Smart Routing system.
            contract.exchange = 'SMART'

            # Set contract primary exchange to help Smart Routing.
            if r[2][:6] == 'NASDAQ':
                contract.primaryExchange = 'ISLAND'
            elif r[2] == 'AMEX':
                contract.primaryExchange = 'AMEX'
            elif r[2] == 'NYSE':
                contract.primaryExchange = 'NYSE'
            else:
                print(r[2])
                assert False
            contract.currency = r[3]

            # Make request to TWS API for this stream, via client.
            self.app.reqMktData(request_id, contract, '', False, False, [])

            # Keep track of request IDs that we are now subscribed to.
            self.watch_list.append((request_id, contract))
            print(f'Subscribed to stream: {r[0]}.')

    def stop_watch(self):
        """
        Gracefully unsubscribe market streams to free up slots for account quota.

        """

        # Unsubscribe to all subscribed market streams.
        for request_item in self.watch_list:
            self.app.cancelMktData(request_item[0])
            print(f'Unsubscribed to stream: {request_item[1].symbol}.')


capturer = StreamListener()


def signal_handler(sig, frame):
    """
    Capture keyboard interrupt signal to cleanly exit.

    """

    print('Ctrl-C pressed.')
    capturer.stop()


# Setup keyboard interrupt signal handler.
signal.signal(signal.SIGINT, signal_handler)

# Start capturing.
capturer.start()
