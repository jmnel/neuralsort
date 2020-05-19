import sqlite3

import settings
from ib_common import *


class TickMessage:

    def __init__(self,
                 timestamp,
                 tick_type,
                 contract_id,
                 price=None,
                 size=None):

        self.timestamp = timestamp
        self.tick_type = tick_type
        self.contract_id = contract_id
        self.price = None
        self.size = None


class BboSimulator:

    def __init__(self, message_mask):

        self.message_mask = message_mask

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME) as db:

            contracts_meta = db.execute(
                'SELECT * FROM contracts_meta;').fetchall()

            ticks_raw = db.execute('''
SELECT * FROM tick_messages ORDER BY timestamp;''').fetchall()

        self.contract_ids, self.symbols, self.sec_type = zip(
            *contracts_meta)

        self.ticks = list()

        for idx, tick_raw in enumerate(ticks_raw):

            _, timestamp, contract_id, tick_type, price_value, size_value = tick_raw

            assert contract_id in self.contract_ids

            if tick_type in self.message_mask:

                self.ticks.append(TickMessage(timestamp,
                                              tick_type,
                                              contract_id,
                                              price_value,
                                              size_value))

    def subscribe(self, msg_queue):

        for idx, tick in enumerate(self.ticks):

            msg_queue.put(tick)
