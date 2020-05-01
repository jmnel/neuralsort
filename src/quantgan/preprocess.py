import signal
import sqlite3
import json
from datetime import datetime

import numpy as np
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
from scipy.stats import kurtosis
from progress.bar import PixelBar
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import MissingDataError

from datasets.eod_close import EodCloseDataset
import lambertw
import settings


def log_returns(s):
    """
    Calculates log returns of a time series.

    """

    x = np.diff(np.log(s))

    s0 = s[0]

    for t in range(len(s)):
        if np.isnan(s[t]) or s[t] == 0:
            print(t)

    return (x, s0)


def normalize(x):
    """
    Normalizes a time series of returns to have mean 0 and unit variance.

    """

    mean = np.mean(x)
    std = np.std(x)

    return ((x - mean) / std, mean, std)


def inv_lambert_w(x):
    return lambertw.gaussianize(x)


class Preprocessor:

    def __init__(self, rebuild=False):
        self.rebuild = rebuild

        self.loader = DataLoader(dataset=EodCloseDataset(),
                                 shuffle=False,
                                 batch_size=1)

        signal.signal(signal.SIGINT, self.exit_cleanly)
        signal.signal(signal.SIGTERM, self.exit_cleanly)

        self.exit_flag = False

    def prep_database(self, db):

        if self.rebuild:

            db.execute('PRAGMA foreign_keys = ON;')
            db.execute('''
DROP TABLE IF EXISTS quantgan_meta;
''')
            db.execute('''
DROP TABLE IF EXISTS quantgan_log_returns;
''')

        db.execute('''
CREATE TABLE IF NOT EXISTS quantgan_meta(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL,
    meta_json TEXT NOT NULL,
    UNIQUE(symbol)
);
''')

        db.execute('''
CREATE TABLE IF NOT EXISTS quantgan_data(
    id INTEGER PRIMARY KEY,
    symbol CHAR(32) NOT NULL,
    date DATE NOT NULL,
    log_return FLOAT NOT NULL,
    x_norm1 FLOAT NOT NULL,
    x_gaus FLOAT NOT NULL,
    x_norm2 FLOAT NOT NULL);
''')

    def get_stock_skip_list(self, db):

        rows = db.execute('SELECT symbol FROM quantgan_meta;').fetchall()
        skip_list = set([row[0] for row in rows])

        return skip_list

    def process(self, db, skip_list):

        bar = PixelBar('Processing', max=len(self.loader),
                       suffix='%(index)d/%(max)d %(eta)ds')

        for idx, record in enumerate(self.loader):

            if self.exit_flag:
                break

            symbol, s, dates = record[0]

            rows = db.execute('''
SELECT symbol FROM quantgan_meta
WHERE symbol = ?;
''', (symbol[0],)).fetchall()
            if len(rows) > 0:
                continue

            if symbol in skip_list:
                continue

            dates = dates[1:]
            dates = [d[0] for d in dates]

            symbol = symbol[0]

            s = s.flatten().numpy()

            x, s_init = log_returns(s)
            x_norm1, mean1, std1 = normalize(x)

            try:
                x_gaus, tau = inv_lambert_w(x_norm1)

            except ValueError as error:
                print(f'Failed to apply Lambert W: {error}')
                print(f'Skipping {symbol}')
                continue

            x_norm2, mean2, std2 = normalize(x_gaus)

            kurtosis1 = kurtosis(x_norm1, fisher=False, bias=False)
            kurtosis2 = kurtosis(x_norm2, fisher=False, bias=False)

            mean3 = np.mean(x_norm2)
            std3 = np.mean(x_norm2)

            try:
                adf = adfuller(x_norm2)

            except MissingDataError as error:
                print(f'Failed to calculate unit root test for {symbol}.')
                continue

            meta = {
                's_init': s_init,
                'mean1': mean1,
                'std1': std1,
                'tau_mu': tau[0],
                'tau_sigma': tau[1],
                'tau_delta': tau[2],
                'mean2': mean2,
                'std2': std2,
                'kurtosis1': kurtosis1,
                'kurtosis2': kurtosis2,
                'mean3': mean3,
                'std3': std3,
                'adf': {
                    'adf_score': adf[0],
                    'pvalue': adf[1],
                    'used_lag': adf[2],
                    'nobs': adf[3],
                    'critical_values': adf[4],
                    'icbest': adf[5]
                }
            }

            meta_json = json.dumps(meta)

            data = [(symbol, dates[i], x[i], x_norm1[i], x_gaus[i], x_norm2[i])
                    for i in range(len(x))]

            db.execute('''
INSERT INTO quantgan_meta(symbol, meta_json)
    VALUES(?, ?);
''', (symbol, meta_json))

            db.executemany('''
INSERT INTO quantgan_data(
    symbol,
    date,
    log_return,
    x_norm1,
    x_gaus,
    x_norm2)
    VALUES(?, ?, ?, ?, ?, ?);
''', data)

            db.commit()

#            for k, v in adf1[4].items():
#                print(f'{k} : {v}')
#            for k, v in adf2[4].items():
#                print(f'{k} : {v}')
            bar.next()
#            exit()

#            print(f'k1: {kurtosis1}, k2: {kurtosis2}')

        bar.finish()

        print('Programing exiting')

    def run(self):

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
            self.prep_database(db)
            skip_list = self.get_stock_skip_list(db)
            self.process(db, skip_list)

    def exit_cleanly(self, signum, frame):

        print(f'\n\nKill signal received. Finishing last batch and cleaning up.')
        self.exit_flag = True


pre = Preprocessor(rebuild=True)
pre.run()


# symbol, s = next(iter(loader))[0]

# s = s[0].numpy()

# x, s0 = log_returns(s)

# x_norm, mean1, std1 = normalize(x)

# ax2 = plt.subplot(2, 2, 2)
# seaborn.distplot(x_norm, bins=200, ax=ax2)

# x_gaus, tau = inv_lambert_w(x_norm)

# x_norm2, mean2, var2 = normalize(x_gaus)

# ax3 = plt.subplot(2, 2, 3)
# seaborn.distplot(x_gaus, bins=200, ax=ax3)

# ax4 = plt.subplot(2, 2, 4)
# seaborn.distplot(x_norm2, bins=200, ax=ax4)

# k1 = kurtosis(x_norm, fisher=False, bias=False)
# k2 = kurtosis(x_norm2, fisher=False, bias=False)

# print(f'k1: {k1}, k2: {k2}')

# plt.show()

# symbol, s = next(iter(loader))

# print(symbol)
