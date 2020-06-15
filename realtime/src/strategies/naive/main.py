import sqlite3
from pprint import pprint
from datetime import datetime, timedelta, time

import numpy as np
import quandl
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
from scipy.stats import gumbel_r

import settings


def historical_fit():
    quandl.ApiConfig.api_key = settings.QUANDL_API_KEY
    res = quandl.get('EOD/GOOGL',
                     start_date='2019-01-01',
                     end_date='2020-05-03')
    df = res.loc[:, ['Adj_High', 'Adj_Open', 'Adj_Close']]
    inv_rets = df.loc[:, 'Adj_High'] / df.loc[:, 'Adj_Open'] - 1.
    df['Inv_Returns'] = inv_rets

    mu, beta = gumbel_r.fit(df['Inv_Returns'])
    print(f'Historical Gumbel fit: μ = {mu}, β = {beta}')
#    seaborn.distplot(df.loc[:, 'Inv_Returns'], bins=200)

    last_close = df.iloc[-1][-2]

    return mu, beta, last_close


def get_messages():
    with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:
        rows = db.execute('''
SELECT * FROM iex_trade_reports WHERE symbol=="GOOGL";
''').fetchall()

    return rows


def simulate_trading(gumbel_params, last_close, messages):
    print('Simulating simple trading strategy...')

    cash = 2000.
    holdings = 0

    past_three = (None,) * 3

    buy_sell_signals = list()
    cash_history = list()
    holdings_history = list()
    price_history = list()
    value_history = list()

    mu_fit, _ = gumbel_params

    for idx, msg in enumerate(messages):

        #        print(msg)

        t = datetime.utcfromtimestamp(msg[2] * 1e-9) - timedelta(hours=4)
#        print(t.strftime('%H:%M:%S'))

        price = msg[4]
        buy_sell_signal = 0

        if holdings == 0:
            before_4pm = t.time() < time(hour=15, minute=58, second=0)

            if before_4pm:

                three_ups = False
                if past_three[0] and past_three[1] and past_three[2]:
                    if (past_three[0] < past_three[1] and
                            past_three[1] < past_three[2] and
                            past_three[2] < price and
                            price < last_close):
                        three_ups = True

                crossing = False
                if past_three[-1]:
                    if past_three[-1] < last_close and last_close < price:
                        crossing = True

                if three_ups:
                    print(f'{idx} : ', end='')
                    print(f'Buy signal @ ${price}: 3 ups')
                    buy_sell_signal = 1
                    holdings = 1
                    cash -= price

                elif crossing:
                    print(f'{idx} : ', end='')
                    print(f'Buy signal @ ${price}: crossing')
                    buy_sell_signal = 1
                    holdings = 1
                    cash -= price

        else:

            after_4pm = t.time() >= time(hour=15, minute=58, second=0)
            limit_down = price < 0.97 * last_close

            five_downs_before = False
            three_downs_after = False
#            if len(price_history) > 0:
#                inv_ret = price / price_history[0] - 1.

            if len(price_history) > 0:
                mu = (mu_fit + 1) * price_history[0]

            if len(price_history) >= 5:
                if(price_history[-5] > price_history[-4] and
                        price_history[-4] > price_history[-3] and
                        price_history[-3] > price_history[-2] and
                        price_history[-2] > price_history[-1] and
                        price_history[-1] > price and
                        price > mu):
                    five_downs_before = True

            if len(price_history) >= 3:
                if(mu > price_history[-3] and
                        price_history[-3] > price_history[-2] and
                        price_history[-2] > price_history[-1] and
                        price_history[-1] > price):
                    three_downs_after = True

            if after_4pm:
                print(f'{idx} : ', end='')
                print(f'Sell signal @ ${price}: near close')
                holdings = 0
                cash += price
                buy_sell_signal = -1

            elif limit_down:
                print(f'{idx} : ', end='')
                print(f'Sell signal @ ${price}: limit down')
                holdings = 0
                cash += price
                buy_sell_signal = -1

            elif five_downs_before:
                print(f'{idx} : ', end='')
                print(f'Sell signal @ ${price}: 5 down before mean')
                holdings = 0
                cash += price
                buy_sell_signal = -1

            elif three_downs_after:
                print(f'{idx} : ', end='')
                print(f'Sell signal @ ${price}: 3 down after mean')
                holdings = 0
                cash += price
                buy_sell_signal = -1

        past_three = (past_three[1], past_three[2], price)

        cash_history.append(cash)
        holdings_history.append(holdings)
        price_history.append(price)
        value_history.append(holdings * price)
        buy_sell_signals.append(buy_sell_signal)

#    plt.plot(np.array(cash_history), linewidth=0.5)
#    plt.plot(np.array(value_history), linewidth=0.5)
    print(f'ending cash: ${cash}')
    print(f'return: ${cash-2000}')

    print(f'hold return: ${price_history[-1] - price_history[0]}')
#    matplotlib.rcParams['figure.dpi'] = 80
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(price_history, linewidth=0.5, color='C2')
    ax[0].set_title('Fig. 1: GOOGL: Price')
    ax[1].plot(np.array(cash_history) + np.array(value_history), linewidth=0.5, color='C1')
    ax[1].set_title('Fig. 2: Portfolio value (cash + holdings)')
    markerline, stemline, _ = ax[2].stem(np.arange(len(buy_sell_signals)), np.array(buy_sell_signals))
    ax[2].set_title('Fig. 3: Buy/sell signal')
    plt.setp(stemline, linewidth=0.5)
    plt.setp(markerline, markersize=0.4)
    plt.tight_layout()
    plt.savefig('test.png', dpi=200)
#    plt.show()


def main():

    mu, beta, last_close = historical_fit()
    print(f'Last close: {last_close}')
    messages = get_messages()
    print(f'Got {len(messages)} messages for GOOGL')

    simulate_trading((mu, beta), last_close, messages)


if __name__ == '__main__':
    main()
