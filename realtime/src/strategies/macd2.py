import logging
import sqlite3
from pprint import pprint
from datetime import datetime, timedelta, time

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

import settings
from strategy import Strategy
from ema import exp_moving_avg

logger = logging.getLogger(__name__)


class MacdStrategy(Strategy):

    def __init__(self,
                 symbol=None,
                 ema1_lag=10,
                 ema2_lag=40,
                 ema1_decay=2,
                 ema2_decay=2,
                 macd_lag=9,
                 macd_decay=2,
                 macd_threshold=1e-3,
                 ):

        super().__init__()

        self.cash = 1.0

        self.ema1_lag = ema1_lag
        self.ema2_lag = ema2_lag
        self.ema1_decay = ema1_decay
        self.ema2_decay = ema2_decay
        self.macd_decay = macd_decay
        self.macd_lag = macd_lag
        self.macd_threshold = 1e-3

        self.buy_signal = 0

        self.symbol = symbol
        self.tick_count = 0

        self.holding = 0
        self.ema1 = None
        self.ema2 = None
        self.macd_ema = None

        self.price_hist = list()
        self.size_hist = list()
        self.ema1_hist = list()
        self.ema2_hist = list()
        self.macd_hist = list()
        self.macd_ema_hist = list()
        self.macd_cross_hist = list()
        self.buy_signal_hist = list()
        self.holdings_hist = list()
        self.cash_hist = list()

    def tick(self,
             timestamp: int,
             price: float,
             size: int):

        logger.debug(f'{self.tick_count} : MACD tick: {price}, {size}')

        self.ema1 = exp_moving_avg(price,
                                   self.ema1,
                                   self.ema1_lag,
                                   self.ema1_decay)
        self.ema2 = exp_moving_avg(price,
                                   self.ema2,
                                   self.ema2_lag,
                                   self.ema2_decay)
        self.macd = self.ema1 - self.ema2
        self.macd_ema = exp_moving_avg(self.macd,
                                       self.macd_ema,
                                       self.macd_lag,
                                       self.macd_decay)
        self.macd_cross = self.macd - self.macd_ema

        t = datetime.utcfromtimestamp(timestamp * 1e-9) - timedelta(hours=5)
        buy_sell_signal = 0
        if self.holding == 0:

            before_4pm = t.time() < time(hour=15, minute=50, second=0)

            if before_4pm:
                macd_cross_up = False
                if self.tick_count > 0:
                    if self.macd_ema_hist[-1] <= 0 and self.macd_ema > 0:
                        #                    if self.macd_cross_hist[-1] <= 0 and self.macd_cross > 0:
                        logger.info(f'{self.tick_count} - {self.symbol} - MACD buy signal: reason MACD-EMA cross up.')
                        self.holding = self.cash / price
                        self.cash = 0
                        self.buy_signal = 1

        else:

            after_4pm = t.time() >= time(hour=15, minute=50, second=0)

            if after_4pm:
                logger.info(f'{self.tick_count} - {self.symbol} - MACD sell: reason after 4pm')
                self.cash = self.holding * price
                self.holding = 0
                self.buy_signal = -1

            elif self.tick_count > 0:
                if price < self.price_hist[0] * 0.97:
                    logger.info(f'{self.tick_count} - {self.symbol} - MACD sell signal: reason limit down 3%')
                    self.cash = self.holding * price
                    self.holding = 0
                    self.buy_signal = -1

#                elif self.macd_cross_hist[-1] > 0 and self.macd_cross <= 0:
                elif self.macd_ema_hist[-1] > 0 and self.macd_ema <= 0:
                    logger.info(f'{self.tick_count} - {self.symbol} - MACD sell signal: reason MACD-EMA cross down')
                    self.cash = self.holding * price
                    self.holding = 0
                    self.buy_signal = -1

        self.price_hist.append(price)
        self.size_hist.append(size)
        self.ema1_hist.append(self.ema1)
        self.ema2_hist.append(self.ema2)
        self.macd_hist.append(self.macd)
        self.macd_ema_hist.append(self.macd_ema)
        self.macd_cross_hist.append(self.macd_cross)

        self.cash_hist.append(self.cash)
        self.holdings_hist.append(self.holding)
        self.buy_signal_hist.append(self.buy_signal)

        self.tick_count += 1

    def notify_close(self):

        if self.holding > 0:
            logger.info(f'CLOSE - {self.symbol} - MACD sell signal: reason close-notify')
            self.cash = self.holding * self.price_hist[-1]
            self.holding = 0
            self.buy_signal = -1

            self.cash_hist.append(self.cash)
            self.holdings_hist.append(self.holding)
            self.buy_signal_hist.append(self.buy_signal)


macd = MacdStrategy(macd_lag=40)

db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME)

days = db.execute('SELECT date FROM iex_days WHERE date >= "2019-01-01" AND date <= "2019-12-31";').fetchall()
days = tuple(zip(*days))[0]
day = days[50]

data = db.execute('SELECT timestamp, price, size FROM iex_trade_reports WHERE day=? AND symbol =?;',
                  (day, 'GOOGL')).fetchall()

db.close()

for idx, message in enumerate(data):
    out = macd.tick(*message)

fig, ax = plt.subplots(3, 1)

ax[0].plot(macd.price_hist, linewidth=0.8)
ax[0].plot(macd.ema1_hist, linewidth=0.8)
ax[0].plot(macd.ema2_hist, linewidth=0.8)

ax[1].plot(macd.macd_hist, linewidth=0.8)
ax[1].plot(macd.macd_ema_hist, linewidth=0.8)
# ax[1].plot(macd.macd_cross_hist)

i0 = 0
ax[2].plot(macd.price_hist, linewidth=0.5, color='C1')

for i in range(len(macd.holdings_hist)):
    i1 = min(i + 1, len(macd.holdings_hist))
    if macd.holdings_hist[i0] == 0 and macd.holdings_hist[i] > 0:
        print(f'sell {i0}:{i}')
        ax[2].plot(np.arange(i0, i1), macd.price_hist[i0: i1], linewidth=0.8, color='C0')
        i0 = i

    elif macd.holdings_hist[i0] > 0 and macd.holdings_hist[i] == 0:
        print(f'buy {i0}:{i}')
        ax[2].plot(np.arange(i0, i1), macd.price_hist[i0: i1], linewidth=0.8, color='C5')
        i0 = i

print(f'{macd.cash_hist[-1]}')
print(f'return: {macd.cash_hist[-1] - macd.cash_hist[0]}')
print(f'hold: {(macd.price_hist[-1] - macd.price_hist[0]) / macd.price_hist[0]}')

plt.show()
