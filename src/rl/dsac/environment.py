from pprint import pprint

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from tick_bar_dataset import TickBarDataset


class Environment:

    def __init__(self, trade_penalty=0.005, invalid_penalty=0.1):
        self.train_loader = DataLoader(TickBarDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
        self.init_len = 10
        self.trade_penalty = trade_penalty

        self.init_cash = 100.0
        self.hold = 0.0

    def reset(self):

        self.idx = self.init_len

        self.buy_pts = list()
        self.sell_pts = list()

#        try:
#            self.day, self.stock, self.x = next(self.train_iter)
#        except StopIteration:
#            self.train_iter = iter(self.train_loader)
#            self.day, self.stock, self.x = next(self.train_iter)

        it = iter(self.train_loader)
        for i in range(6):
            next(it)
        self.day, self.stock, self.ticks = next(it)


#        self.prices = list(0.5 * (t[0,2] + t[5]) for t in self.ticks)

        self.prices = np.zeros(self.ticks.shape[1])
        for i in range(len(self.prices)):
            self.prices[i] = 0.5 * (self.ticks[0, i, 2] + self.ticks[0, i, 5])

        alpha = 2
        l1 = 20
        ema1 = np.zeros_like(self.prices)
        for t in range(len(self.prices)):
            ema1[t] = self.prices[t] * (alpha / (1 + l1))
            if t == 0:
                ema1[t] = self.prices[0]
#                ema1[t] += self.prices[0]
            else:
                ema1[t] += ema1[t - 1] * (1 - alpha / (1 + l1))

        self.ema1 = ema1

        #state = self.ticks[:, self.idx - self.init_len + 1: self.idx + 1, :]
#        print(state.shape)

        self.actions = list(0.0 for _ in range(self.init_len))
        self.rewards = list(0.0 for _ in range(self.init_len))

        self.net = 0

        actions_t = torch.FloatTensor(self.actions[-1:])
        rewards_t = torch.FloatTensor(self.rewards[-self.init_len:])
        rewards_t = rewards_t.reshape((1, rewards_t.shape[0], 1))
        state = (rewards_t, actions_t)
#        state = (state, actions_t)
        return state

    def step(self, action):

        # Flat action.
        if action == 0:
            reward = 0 - self.trade_penalty * np.abs(self.actions[-1] - action)
#            reward = (self.ema1[self.idx] - self.ema1[self.idx - 1]
#                      - self.trade_penalty * np.abs(self.actions[-1] - action))
            if action != self.actions[-1]:
                self.sell_pts.append((self.idx, self.prices[self.idx]))

        # Long action
        elif action == 1:
            #            reward = (self.prices[self.idx] - self.prices[self.idx - 1]
            #                      - self.trade_penalty * np.abs(self.actions[-1] - action))
            reward = (self.ema1[self.idx] - self.ema1[self.idx - 1]
                      - self.trade_penalty * np.abs(self.actions[-1] - action))
            if action != self.actions[-1]:
                self.buy_pts.append((self.idx, self.prices[self.idx]))

        self.net += reward

        self.rewards.append(action)

        self.idx += 1
        self.actions.append(action)
        done = self.idx + 1 >= self.ticks.shape[1]

        state = self.ticks[:, self.idx - self.init_len + 1: self.idx + 1, :]
        actions_t = torch.FloatTensor(self.actions[-1:])
        rewards_t = torch.FloatTensor(self.rewards[-self.init_len:])
        rewards_t = rewards_t.reshape((1, rewards_t.shape[0], 1))
        state = (rewards_t, actions_t)
        return state, reward, done

#        return (state, torch.FloatTensor(self.actions[-1])), reward, done
