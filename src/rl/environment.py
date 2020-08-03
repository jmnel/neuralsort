from pprint import pprint

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from ticks_dataset import TicksDataset, TicksDatasetIEX


class Environment:

    def __init__(self, trade_penalty=0.005):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
        self.init_len = 3
        self.trade_penalty = 0.005

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

        p = self.x[0, :, 1].numpy()
        p = p[:200]
        self.prices = p

        p = p / p[0]
        p = np.diff(np.log(p)) * 1.0e2
        self.x = torch.FloatTensor(p)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        self.p_rewards = list()
        self.r_rewards = list()

        self.hold_len = 0.0
        self.max_hold_len = 0.0

        state = self.x[:, :self.idx, :]

        state = torch.cat((state, torch.zeros_like(state)), dim=-1)

        return state

    def step(self, action):
        self.actions.append(action)

        reward = 0.0

        p = self.prices[self.idx]

        if self.idx + 1 < self.x.shape[1]:
            r = self.x[0, self.idx + 1, 0] * 1e-1
        else:
            r = 0.0

#        pprint(self.x[0, self.idx, 0])

        if action == 0:

            self.flat_count += 1

            # Sell point.
            if self.hold:
                self.hold = False
                self.sell_pts.append((self.idx, p))
                reward = p - self.trade_penalty + self.hold_len
                self.hold_len = 0
                self.net += p - self.trade_penalty
                self.p_rewards.append(reward)

            reward -= r
            self.r_rewards.append(-r)

        else:

            self.long_count += 1

            # Buy point.
            if not self.hold:
                self.hold = True
                self.buy_pts.append((self.idx, p))
                reward = -p - self.trade_penalty
                self.buy_price = p
                self.net += -p - self.trade_penalty
                self.p_rewards.append(reward)
            else:
                self.hold_len += 1.0

            reward += r
            self.r_rewards.append(r)

        self.max_hold_len = max(self.max_hold_len, self.hold_len)

        self.idx += 1
        state = self.x[:, :self.idx, :]

        s_actions = torch.FloatTensor(self.actions)
        s_actions = torch.cat((torch.zeros(3), s_actions), dim=0)
        s_actions = s_actions.reshape_as(state)
        state = torch.cat((state, s_actions), dim=-1)

        done = self.idx >= self.x.shape[1]

        if done and self.hold:
            self.hold = False
            self.sell_pts.append((self.idx, p))
            reward = p - self.trade_penalty
            self.net += p - self.trade_penalty
            self.p_rewards.append(reward)

        return state, reward, done
