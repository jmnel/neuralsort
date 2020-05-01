from datetime import datetime
import operator
import sqlite3
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

import settings


def show_double_listed(db, symbol_id, symbol_name):

    data = db.execute('''
SELECT
    eod_exchanges.name,
    eod_ohlcv.date,
    eod_ohlcv.open,
    eod_ohlcv.high,
    eod_ohlcv.low,
    eod_ohlcv.close,
    eod_ohlcv.volume
FROM eod_ohlcv
INNER JOIN eod_exchanges ON eod_ohlcv.exchange_id=eod_exchanges.id
WHERE symbol_id = ?;
''', (symbol_id,)).fetchall()

    exchanges = {row[0] for row in data}

#    fig, axs = plt.subplots(len(exchanges), 1)

    for idx, exchange in enumerate(exchanges):

        filtered = list(filter(lambda row: row[0] == exchange,
                               data))

        filtered = sorted(filtered, key=operator.itemgetter(1))
        _, dates, o, h, l, c, v = zip(*filtered)

        dates = [datetime.strptime(
            d.split(' ')[0], '%Y-%m-%d').date() for d in dates]

        ohlcv = list([(dates[i], o[i]) for i in range(len(dates))])

#        ohlcv = sorted(ohlcv)

        dates, o = zip(*ohlcv)
#        dates, o, h, l, c, v = zip(*ohlcv)

        plt.plot(dates, o, lw=0.4)

        print(f'{symbol_name} on {exchange}')
        print(f'{dates[0]} --> {dates[-1]}\n')

    plt.show()


def main():

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

        exchange_ids, exchange_names = zip(*db.execute('''
SELECT id, name FROM eod_exchanges;
''').fetchall())

        join_table = (db.execute('''
SELECT symbol_id, exchange_id FROM eod_exchange_symbol;
''').fetchall())

        exchange_map = {id: list() for id in exchange_ids}

        symbols = dict(db.execute('''
SELECT * FROM eod_symbols;
'''))

        for sym, exch in join_table:
            exchange_map[exch].append(sym)

        for sym_id, sym_name in symbols.items():

            listed_on = list(filter(lambda ex: sym_id in exchange_map[ex],
                                    exchange_ids))

            if len(listed_on) > 1:
                show_double_listed(db, sym_id, sym_name)


main()
