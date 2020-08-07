from pprint import pprint
import random

import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

from tick_bar_dataset import TickBarDataset


class ActionSpace:

    def sample(self):
        return random.randint(0, 1)


class StateSpace:

    def __init__(self, state_space):
        self.shape = [state_space, ]


class TickEnvironment:

    def __init__(self, trade_penalty=0.05, invalid_penalty=0.1):
        self.train_loader = DataLoader(TickBarDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=True)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
#        self.init_len = 50
        self.trade_penalty = trade_penalty

        self.action_space = ActionSpace()
        self.state_space = StateSpace(50)

    def reset(self):

        try:
            self.day, self.stock, self.ticks = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            self.day, self.stock, self.ticks = next(self.train_iter)


#        it = iter(self.train_loader)
#        for i in range(6):
#            next(it)
#        self.day, self.stock, self.ticks = next(it)

        self.prices = np.zeros(self.ticks.shape[1])
        for i in range(len(self.prices)):
            self.prices[i] = 0.5 * (self.ticks[0, i, 2] + self.ticks[0, i, 5])

#        self.idx = self.init_len
        self.idx = 0

        self.buy_pts = list()
        self.sell_pts = list()
        self.hold_intervals = list()
        self.flat_intervals = list()
        self.last_switch = 0

#        self.actions = list(0.0 for _ in range(self.init_len))
#        self.rewards = list(0.0 for _ in range(self.init_len))

        self.actions = list()
        self.rewards = list()

        self.net = 0

#        p = np.diff(np.log(self.prices[self.idx - self.init_len: self.idx + 1])) * 10.0
        self.data = np.diff(np.log(self.prices))
        n = self.state_space.shape[0]
        self.data = np.concatenate((np.zeros(n), self.data))

        p = self.data[self.idx: self.idx + n]

#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1] / self.prices[0]
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1]
#        state = p
        state = np.concatenate((p, (0,)))
        return state

    def step(self, action):

        prev_action = 0 if len(self.actions) < 1 else self.actions[-1]

        # Flat action.
        if action == 0:
            reward = 0 - self.trade_penalty * np.abs(prev_action - action)
#            reward = -self.prices[self.idx] - self.prices[self.idx - 1]

            if action != prev_action:
                self.sell_pts.append((self.idx, self.prices[self.idx]))
                self.hold_intervals.append((self.last_switch, self.idx))
                self.last_switch = self.idx

        # Long action
        elif action == 1:
            reward = (self.prices[self.idx] - self.prices[self.idx - 1]
                      - self.trade_penalty * np.abs(prev_action - action))
            if action != prev_action:
                self.buy_pts.append((self.idx, self.prices[self.idx]))
                self.flat_intervals.append((self.last_switch, self.idx))
                self.last_switch = self.idx

        self.net += reward

        self.rewards.append(action)

        self.idx += 1
        self.actions.append(action)
        done = self.idx + 1 >= self.prices.shape[0]

        if done and action == 0:
            self.flat_intervals.append((self.last_switch, self.idx))
        if done and action == 1:
            self.hold_intervals.append((self.last_switch, self.idx))

        n = self.state_space.shape[0]
        p = self.data[self.idx: self.idx + n]

#        p = np.diff(np.log(self.prices[self.idx - self.init_len: self.idx + 1])) * 10.0
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1] / self.prices[0]
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1]
        state = p
        state = np.concatenate((p, (action,)))
        return state, reward, done

    def __len__(self):
        return self.prices.shape[0]
