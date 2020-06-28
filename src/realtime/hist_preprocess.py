import sqlite3
import csv
import requests
import io
from pprint import pprint

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
import quandl
from scipy.stats import gumbel_r
import numpy as np

import settings

quandl.ApiConfig.api_key = settings.QUANDL_API_KEY


def get_meta_data(db):

    rows = db.execute('SELECT * FROM contracts_meta;').fetchall()
    ids, symbols, sec_type = zip(*rows)

    return ids, symbols, sec_type


def prepare_database(db):

    db.execute('DROP TABLE IF EXISTS gumbel_fit;')

    db.execute('''
CREATE TABLE gumbel_fit(
    symbol_id MEDIUM INT UNSIGNED PRIMARY KEY,
    param_mu FLOAT NOT NULL,
    param_beta FLOAT NOT NULL,
    FOREIGN KEY(symbol_id) REFERENCES contracts_meta(id)
);''')


def estimate_historical_fit(symbols, db):

    qdl_codes = list(s.replace(' ', '_') for s in symbols[1:])

    qdl_codes[qdl_codes.index('ARNC')] = 'ARNC_'
#    qdl_codes[qdl_codes.index('BF.B')] = 'BF_B'

    hist_data = list()

    for idx, code in enumerate(qdl_codes):

        print(f'{idx} : Getting EOD/{code}...')

        res = quandl.get('EOD/' + code,
                         start_date='2018-01-01',
                         #                     start_date='2015-01-01',
                         end_date='2018-06-03')

        df = res.loc[:, ['Adj_High', 'Adj_Open']]

        # Check for problems.
#        df_np = df.to_numpy()

#        if idx == 50:
#            pprint(df_np)
#            pprint(np.isfinite(df_np))

#        assert(np.all(np.isfinite(df.loc[:, 'Adj_Open'])))

        inv_rets = df.loc[:, 'Adj_High'] / df.loc[:, 'Adj_Open'] - 1.

        df['Inv_Returns'] = inv_rets

        mu, beta = gumbel_r.fit(df['Inv_Returns'])

        hist_data.append((idx + 1, mu, beta))

    db.executemany('''
INSERT INTO gumbel_fit(symbol_id, param_mu, param_beta)
VALUES(?, ?, ?)
;''', hist_data)

    db.commit()


def main():

    with sqlite3.connect(settings.DATA_DIRECTORY / settings.IB_DATABASE_NAME) as db:

        prepare_database(db)
        ids, symbols, sec_types = get_meta_data(db)

        estimate_historical_fit(symbols, db)


main()

gumb = gumbel_r.fit(log_rets)


x = np.linspace(-0.01, 0.05, 100)
plt.plot(x, gumbel_r.pdf(x, mu, beta))

plt.show()
#    exit()

#    pprint(df[:, 'Adj_High'])
#        print(df)
