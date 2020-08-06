from pprint import pprint

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from ticks_dataset import TicksDataset, TicksDatasetIEX

NO_ACTION_PENALTY = 1e-2
TRADE_REWARD = 0.0
SWITCH_CONST = 0.01
# SWITCH_DECAY = 0.2


class Environment:

    def __init__(self, trade_penalty=0.005, invalid_penalty=0.1):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
        self.init_len = 5
        self.trade_penalty = trade_penalty

        self.init_cash = 100.0
        self.hold = 0.0

    def reset(self):

        self.past_run = self.idx

        self.idx = self.init_len

#        try:
#            self.day, self.stock, self.x = next(self.train_iter)
#        except StopIteration:
#            self.train_iter = iter(self.train_loader)
#            self.day, self.stock, self.x = next(self.train_iter)

        self.day, self.stock, self.x = next(iter(self.train_loader))

        self.actions = list()
        self.buy_pts = list()
        self.sell_pts = list()

        self.cash = self.init_cash
        self.hold = 0

        self.hold_intervals = list()
        self.flat_intervals = list()

        self.invalid_count = 0

        self.hold_hist = [self.hold for _ in range(self.init_len)]
        self.cash_hist = [self.cash for _ in range(self.init_len)]
        self.values = [self.cash, ]

        self.last_flat = 0
        self.last_hold = None

        self.value = 0

        p = self.x[0, :, 1].numpy()
        s = self.x[0, :, 2].numpy()

        alpha = 2
        l1 = 100
        l2 = 200
        ema1 = np.zeros_like(p)
        ema2 = np.zeros_like(p)
        for t in range(len(p)):
            ema1[t] = p[t] * (alpha / (1 + l1))
            ema2[t] = p[t] * (alpha / (1 + l2))
            if t == 0:
                ema1[t] += p[0]
                ema2[t] += p[0]
            else:
                ema1[t] += ema1[t - 1] * (1 - alpha / (1 + l1))
                ema2[t] += ema2[t - 1] * (1 - alpha / (1 + l2))

        macd = ema1 - ema2

        self.ema1 = ema1
        self.ema2 = ema2
        self.macd = macd

        #        s = s[:200]
        #        p = p[:200]

        self.prices = p

#        p = np.diff(np.log(p))

#        p = p * 100
#        p = p / p[1] - 1.0
#        p -= np.mean(p)
#        p = p / np.std(p)

        self.x = torch.FloatTensor(p)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        t0 = self.idx - self.init_len
        t1 = self.idx
        state = self.x[:, t0:t1, :]

        x_ema1 = torch.FloatTensor(ema1[t0:t1])
        x_ema2 = torch.FloatTensor(ema2[t0:t1])
        x_macd = torch.FloatTensor(macd[t0:t1])
        x_ema1 = x_ema1.reshape_as(state)
        x_ema2 = x_ema2.reshape_as(state)
        x_macd = x_macd.reshape_as(state)

        x_c = torch.FloatTensor(self.cash_hist[t0:t1]).reshape_as(state)
        x_h = torch.FloatTensor(self.hold_hist[t0:t1]).reshape_as(state)

        state = torch.cat((state, x_ema1, x_ema2, x_macd, x_c, x_h), dim=-1)
#        state_c = torch.FloatTensor(self.cash_hist).reshape_as(state)
#        state_h = torch.FloatTensor(self.hold_hist).reshape_as(state)

#        state = torch.cat((state, state_c, state_h), dim=-1)

        return state

    def step(self, action):

        self.actions.append(action)

        done = False

        p = self.prices[self.idx]

        prev_value = self.cash + self.hold * p

        reward =  # 0

        if self.idx + 2 >= self.x.shape[1] and self.hold:
            action = 1

        # Do nothing.
        if action == 0:
            pass
#            reward -= last_switch_penalty
#            print(last_switch_penalty)

        # Sell
        elif action == 1:

            if self.hold > 0:
                self.cash += self.hold * p - self.trade_penalty
                self.hold = 0
                self.sell_pts.append((self.idx, p))
                self.last_flat = self.idx

                self.hold_intervals.append((self.last_hold, self.idx))

                new_value = self.cash + self.hold * p
#                reward += TRADE_REWARD
                self.last_switch = self.idx
            else:
                self.invalid_count += 1
                reward = -1
                done = True
#                reward -= last_switch_penalty

        # Buy
        elif action == 2:

            if self.cash > 0 and self.hold <= 0:
                self.hold += (self.cash - self.trade_penalty) / p
                self.cash = 0.0
                self.buy_pts.append((self.idx, p))

                self.last_hold = self.idx

                self.flat_intervals.append((self.last_flat, self.idx))

                new_value = self.cash + self.hold * p
#                reward += TRADE_REWARD
                self.last_switch = self.idx
            else:
                self.invalid_count += 1
                done = True
                reward = -1
#                reward -= last_switch_penalty

        new_value = self.cash + (self.hold * p - self.trade_penalty)
        delta = (new_value - prev_value)
        reward = delta * 1.0

#        if self.idx > self.past_run:
#            reward += 1e-5 * (self.idx - self.past_run)

        self.idx += 1

        if not done:
            done = self.idx + 1 >= self.x.shape[1]

#        if done:
#            if self.idx > self.past_run:
#                reward += self.past_run - self.idx

        if self.hold and done:
            self.hold_intervals.append((self.last_hold, self.idx))

        if not self.hold and done:
            self.flat_intervals.append((self.last_flat, self.idx))

#        if done:
#            reward += (self.cash + self.hold * p - 1.0) * 0.1

#        if done and len(self.buy_pts) < 3:
#            reward -= 0.01 * torch.abs(torch.randn(1))

        self.hold_hist.append(self.hold)
        self.cash_hist.append(self.cash)

        t0 = self.idx - self.init_len
        t1 = self.idx
        state = self.x[:, t0:t1, :]

        x_ema1 = torch.FloatTensor(self.ema1[t0:t1])
        x_ema2 = torch.FloatTensor(self.ema2[t0:t1])
        x_macd = torch.FloatTensor(self.macd[t0:t1])
        x_ema1 = x_ema1.reshape_as(state)
        x_ema2 = x_ema2.reshape_as(state)
        x_macd = x_macd.reshape_as(state)

        x_c = torch.FloatTensor(self.cash_hist[t0:t1]).reshape_as(state)
        x_h = torch.FloatTensor(self.hold_hist[t0:t1]).reshape_as(state)

        state = torch.cat((state, x_ema1, x_ema2, x_macd, x_c, x_h), dim=-1)
#        state = torch.cat((state, state_c, state_h), dim=-1)

        self.value = self.cash + self.hold * p - self.trade_penalty

        return state, reward, done
