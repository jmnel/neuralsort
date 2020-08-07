from pprint import pprint
import random

import torch
import numpy as np
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt


class ActionSpace:

    def sample(self):
        return random.randint(0, 1)


class EasyEnvironment:

    def __init__(self, trade_penalty=0.005, invalid_penalty=0.1):

        self.idx = 0
        self.init_len = 10
        self.trade_penalty = trade_penalty

        t = np.linspace(0, 10, 400) + np.random.randn(1) * 40
        self.prices = np.cos(t * 4) * (1 - 0.05 * np.cos(10 * t)) + 0.001 * np.random.randn(len(t))
        self.t = np.arange(len(self.prices))

        self.action_space = ActionSpace()

    def reset(self):

        t = np.linspace(0, 10, 400) + np.random.randn(1) * 40
        self.prices = np.cos(t * 4) * (1 - 0.05 * np.cos(10 * t)) + 0.01 * np.random.randn(len(t))
        self.prices += 0.5 * np.random.rand(1) * t
        self.t = np.arange(len(self.prices))

        self.idx = self.init_len

        self.buy_pts = list()
        self.sell_pts = list()
        self.hold_intervals = list()
        self.flat_intervals = list()
        self.last_switch = 0

        self.actions = list(0.0 for _ in range(self.init_len))
        self.rewards = list(0.0 for _ in range(self.init_len))

        self.net = 0

        p = np.diff(self.prices[self.idx - self.init_len: self.idx + 1])
        state = p
        return state

    def step(self, action):

        # Flat action.
        if action == 0:
            reward = 0 - self.trade_penalty * np.abs(self.actions[-1] - action)
            if action != self.actions[-1]:
                self.sell_pts.append((self.idx, self.prices[self.idx]))
                self.hold_intervals.append((self.last_switch, self.idx))
                self.last_switch = self.idx

        # Long action
        elif action == 1:
            reward = (self.prices[self.idx] - self.prices[self.idx - 1]
                      - self.trade_penalty * np.abs(self.actions[-1] - action))
            if action != self.actions[-1]:
                self.buy_pts.append((self.idx, self.prices[self.idx]))
                self.flat_intervals.append((self.last_switch, self.idx))
                self.last_switch = self.idx

        self.net += reward

        self.rewards.append(action)

        self.idx += 1
        self.actions.append(action)
        done = self.idx + 1 >= self.prices.shape[0]

#        if done:

        p = np.diff(self.prices[self.idx - self.init_len: self.idx + 1])
#        p = self.prices[self.idx - self.init_len + 1: self.idx + 1]
        state = p
        return state, reward, done

    def __len__(self):
        return self.prices.shape[0] - self.init_len
