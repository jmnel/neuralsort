import logging
import sqlite3
from pprint import pprint
from operator import itemgetter

import numpy as np

import settings
from macd import MacdStrategy

logger = logging.getLogger(__name__)


class StrategyRunner:

    def __init__(self,
                 k=20,
                 mode='topk_true'):

        self.k = 20
        self.get_days()
        self.days = self.days[:2]
        self.get_top_k()

        logger.info('Intraday strategy runner initialized.')

        self.macd_ema1_lag = 10
        self.macd_ema2_lag = 40
        self.macd_ema1_decay = 2
        self.macd_ema2_decay = 2
        self.macd_macd_lag = 9
        self.macd_macd_decay = 9
        self.macd_threshold = 1e-3

    def get_days(self):
        with sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME) as iex_db:
            self.days = tuple(
                zip(*iex_db.execute('SELECT date FROM iex_days WHERE date > "2017-12-31";').fetchall()))[0]

    def get_top_k(self):
        self.day_top_k = list()

        with sqlite3.connect(settings.DATA_DIRECTORY / settings.QDL_DATABASE_NAME) as qdl_db:
            for idx, d in enumerate(self.days):
                logger.info(f'Getting EOD for day {idx+1} of {len(self.days)} : {d}.')
                eod_day = qdl_db.execute('SELECT symbol, high, open FROM qdl_eod WHERE date =?',
                                         (d,)).fetchall()

                for jdx, (symbol, h, o) in enumerate(eod_day):
                    assert o > 0.
                    eod_day[jdx] = (symbol, h, o, h / o - 1.)

                eod_day = sorted(eod_day, key=itemgetter(3), reverse=True)[:self.k]
                self.day_top_k.append(eod_day)

    def run(self):
        logger.info(f'Backtesting intraday strategy over {len(self.days)} days.')

        for day_idx, day in enumerate(self.days):
            logger.info(f'Backtesting day {day_idx+1} of {len(self.days)}.')
            logger.info(f'True top-{self.k}:')
            for rank_idx, (symbol, o, h, ho) in enumerate(self.day_top_k[day_idx]):
                logger.info(f'  {rank_idx+1} - {symbol}: {o}, {h}, {ho}')

            for rank_idx, (symbol, o, h, ho) in enumerate(self.day_top_k[day_idx]):

                macd_instance = MacdStrategy(symbol,
                                             self.macd_ema1_lag,
                                             self.macd_ema2_lag,
                                             self.macd_ema1_decay,
                                             self.macd_ema2_decay,
                                             self.macd_macd_lag,
                                             self.macd_macd_decay,
                                             self.macd_threshold,
                                             )

                db = sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME)

                messages = db.execute('SELECT timestamp, price, size FROM iex_trade_reports WHERE day=? AND symbol=?;',
                                      (day, symbol)).fetchall()

                if len(messages) < 10:
                    print(f'  {symbol} has insufficient messages')
                    continue
#                assert len(messages) > 10

                for msg in messages:
                    macd_instance.tick(*msg)
                macd_instance.notify_close()

                assert macd_instance.holdings_hist[-1] == 0

                ret = macd_instance.cash_hist[-1] - macd_instance.cash_hist[0]
                print(f'  {symbol}: {ret}')


#            logger.info(f'Getting messages for {day}')

#            with sqlite3.connect(settings.DATA_DIRECTORY / settings.IEX_DATABASE_NAME) as iex_db:
#                messages = iex_db.execute('SELECT * FROM iex_trade_reports WHERE day=?;',
#                                          (day,)).fetchall()
#            logger.info(f'Got {len(messages)}')

            print('\n')


foo = StrategyRunner(k=20)
foo.run()
