from pprint import pprint
import sqlite3
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
# from scipy.stats import linregress
import scipy.signal

import settings


def prepare_database(db):

    symbols = db.execute('''SELECT id, symbol FROM eod_symbols;''').fetchall()
#    print(len(symbols))

    symbols = symbols[1130:1160]

    for idx, (symbol_id, symbol) in enumerate(symbols):

        ohlcv = db.execute('''
SELECT date, open, high, low, close, volume FROM eod_ohlcv
   WHERE symbol_id=?;
''', (symbol_id,)).fetchall()

        # Convert to numpy array.
        dates = [datetime.strptime(
            row[0].split(' ')[0], '%Y-%m-%d').date() for row in ohlcv]

        o = np.array([row[1] for row in ohlcv])
        h = np.array([row[2] for row in ohlcv])
        l = np.array([row[3] for row in ohlcv])
        c = np.array([row[4] for row in ohlcv])
        v = np.array([row[5] for row in ohlcv])

        ohlcv = list(zip(dates, o, h, l, c, v))
        ohlcv = sorted(ohlcv)[:-150]
        ohlcv = zip(*ohlcv)

        dates, o, h, l, c, v = ohlcv
        t = np.array([(d - dates[0]).days for d in dates])
        fig, axs = plt.subplots(3, 1)

        axs[0].plot(dates, o, lw=0.4)

        # Take log and detrend data.
        o, h, l, c = np.log(o), np.log(h), np.log(l), np.log(c)
        o_lin = scipy.signal.detrend(o, type='linear')
        axs[1].plot(dates, o_lin, lw=0.4)

#        w = scipy.signal.blackman(len(o))
#        w = scipy.signal.hann(50)
#        freq = scipy.fft(scipy.convolve(w, o_lin))

#        freq = np.log((freq**2) / np.max(freq))

#        axs[3].plot(freq, lw=0.4)
#        axs[3].set_ylim((0, 5e-1))
#        axs[3].set_xlim((0, 1000))

        # Subtract mean and divide by strandard of deviation.
        o_mu = np.mean(o_lin)
        o_std = np.std(o_lin)

        o_norm = 0.5 * (o_lin - o_mu) / o_std

        axs[2].plot(dates, o_norm, lw=0.4)

        fig.suptitle(symbol)
        plt.show()


def main():

    # Connect to database; create it if it doesn't exist already.
    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

        prepare_database(db)


main()
