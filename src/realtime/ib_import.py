import os
import pickle
from pathlib import Path
from pprint import pprint
import sqlite3
from datetime import datetime

import ibapi

import settings
from ib_common import IB_TICK_TYPES

data_path = settings.DATA_DIRECTORY / 'ib'


def prepare_database(db: sqlite3.Connection):

    db.execute('PRAGMA foreign_keys = ON;')
    db.execute('DROP TABLE IF EXISTS gumbel_fit;')
    db.execute('DROP TABLE IF EXISTS tick_messages;')
    db.execute('DROP TABLE IF EXISTS contracts_meta;')

    db.execute('''
CREATE TABLE contracts_meta(
    id INTEGER PRIMARY KEY,
    symbol CHAR(16) NOT NULL,
    security_type CHAR(8) NOT NULL
);''')

    db.execute('''
CREATE TABLE tick_messages(
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    contract_id MEDIUM INT UNSIGNED NOT NULL,
    message_type TINYINT NOT NULL,
    price_value FLOAT,
    size_value INTEGER,
    FOREIGN KEY(contract_id) REFERENCES contracts_meta(id)
);''')


def parse_ib_bbo(db: sqlite3.Connection, paths):

    message_types = set()
    contract_meta = dict()
    ticks_data = list()

    for file_idx, path in enumerate(paths):
        with open(path, 'rb') as data_file:
            raw_data = pickle.load(data_file)

            for row in raw_data['reqMktData']:
                req_id = row['reqId']
                symbol = row['contract'].symbol
                sec_type = row['contract'].secType

                if req_id not in contract_meta:
                    contract_meta[req_id] = {
                        'req_id': req_id,
                        'symbol': symbol,
                        'sec_type': sec_type}

                    db.execute('INSERT INTO contracts_meta VALUES(?, ?, ?);',
                               (req_id, symbol, sec_type))
                else:
                    assert req_id == contract_meta[req_id]['req_id']
                    assert symbol == contract_meta[req_id]['symbol']
                    assert sec_type == contract_meta[req_id]['sec_type']

            # Parse price type tick messages.
            for row in raw_data['tickPrice']:
                message_types.add(row['tickType'])

                ticks_data.append((row['timestamp'],
                                   row['reqId'],
                                   row['tickType'],
                                   row['price'],
                                   0))

#                if row['reqId'] not in contract_meta:
#                    print(row['reqId'])
#                assert row['reqId'] in contract_meta

            # Parse size type tick messages.
            for row in raw_data['tickSize']:
                message_types.add(row['tickType'])

                ticks_data.append((row['timestamp'],
                                   row['reqId'],
                                   row['tickType'],
                                   0,
                                   row['size']))
#                if row['reqId'] not in contract_meta:
#                    print(row['reqId'])
#                assert row['reqId'] in contract_meta

    db.commit()

    db.executemany('''INSERT INTO tick_messages(
            timestamp,
            contract_id,
            message_type,
            price_value,
            size_value) VALUES(?, ?, ?, ?, ?);
''', ticks_data)

    db.commit()

    print(message_types)
    exit()


#        pprint(raw_data.keys())
#        pprint(raw_data['tickPrice'][0])

#        price_ticks = raw_data['tickPrice']
#        size_ticks = raw_data['tickSize']
#        price_req_ids = {t['reqId'] for t in price_ticks}
#        size_req_ids = {t['reqId'] for t in size_ticks}

#        print(raw_data.keys())

#        if 'reqMktData' in raw_data:
    #            pprint(raw_data['reqMktData'])
#            for row in raw_data['reqMktData']:
#                req_id = row['reqId']
#                symbol = row['contract'].symbol
#                sec_type = row['contract'].secType
#                exit()
#                if req_id not in contracts:
#                    row[req_id] = {'req_id': req_id,
#                                   'symbol': symbol,
#                                   'sec_type': sec_type}

#                else:
#                    assert row[req_id]['req_id'] == req_id
#                    assert row[symbol]['symbol'] == symbol
#                    assert row[sec_type]['sec_type'] == sec_type
#                    print(f'symbol {symbol} appears more than once.')

#        print(day_str)
#        assert price_req_ids == size_req_ids
#        d1 = price_req_ids.difference(size_req_ids)
#        d2 = size_req_ids.difference(price_req_ids)

#        if len(d1) != 0:
#            print(d1)
#        if len(d2) != 0:
#            print(d2)

#        for idx, tick in enumerate(size_ticks):

#            types.add(tick['tickType'])

#            if tick['tickType'] == 66:
#                pprint(tick)

#    exit()

#    pprint(types)


def main():

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME) as db:

        prepare_database(db)

        files = list()
        for idx, fname in enumerate(os.listdir(data_path)):
            fpath = data_path / fname

            if fpath.suffix == '.p':
                files.append(fpath)

        parse_ib_bbo(db, files)


main()
