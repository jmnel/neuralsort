from pprint import pprint

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from ticks_dataset import TicksDataset, TicksDatasetIEX


class Environment:

    def __init__(self, trade_penalty=0.005, invalid_penalty=0.1):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
        self.init_len = 3
        self.trade_penalty = 0.005
        self.invalid_penalty = invalid_penalty

    def reset(self):
        self.idx = self.init_len
        self.day, self.stock, self.x = next(iter(self.train_loader))

        self.flat_count = 0
        self.long_count = 0

        self.actions = list()
        self.buy_pts = list()
        self.sell_pts = list()

        self.hold = False
        self.buy_price = None
        self.net = 0

        self.hold_intervals = list()
        self.flat_intervals = list()

        p = self.x[0, :, 1].numpy()
        p = p[:200]
        self.prices = p

        p = p / p[0]
        p = np.diff(np.log(p)) * 1.0e2
        self.x = torch.FloatTensor(p)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        self.invalid_count = 0
        self.hold_count = 0

        state = self.x[:, :self.idx, :]

        state = torch.cat((state, torch.zeros_like(state)), dim=-1)

        self.hold_raw = 0
        self.sell_raw = 0
        self.buy_raw = 0

        if self.hold:
            hold_f = 1.0
        else:
            hold_f = 0.0

        return (state, hold_f)

    def step(self, action):
        self.actions.append(action)

        reward = 0.0

        p = self.prices[self.idx]

        # Do nothing.
        if action == 0:
            #            reward = self.invalid_penalty
            self.hold_count += 1
            self.hold_raw += 1

        # Sell
        elif action == 1:
            self.sell_raw += 1
            if not self.hold:
                reward = -self.invalid_penalty
                self.invalid_count += 1
            else:
                reward = p + self.trade_penalty + self.invalid_penalty
                self.sell_pts.append((self.idx, p))
                self.net += p - self.trade_penalty
                self.hold = False

        # Buy
        elif action == 2:
            self.buy_raw += 1
            if self.hold:
                reward = -self.invalid_penalty
                self.invalid_count += 1
            else:
                reward = -p - self.trade_penalty + self.invalid_penalty
                self.buy_pts.append((self.idx, p))
                self.net += -p - self.trade_penalty
                self.hold = True

        self.idx += 1
        done = self.idx >= self.x.shape[1]

        if done and self.hold:
            reward = p - self.invalid_penalty - self.trade_penalty
            self.invalid_count += 1
            self.sell_pts.append((self.idx - 1, p))
            self.net += p - self.trade_penalty

        state = self.x[:, :self.idx, :]

        s_actions = torch.FloatTensor(self.actions)
        s_actions = torch.cat((torch.zeros(3), s_actions), dim=0)
        s_actions = s_actions.reshape_as(state)
        state = torch.cat((state, s_actions), dim=-1)

        if self.hold:
            hold_f = 1.0
        else:
            hold_f = 0.0

        return (state, hold_f), reward, done
