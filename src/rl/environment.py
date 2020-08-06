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
        self.init_len = 0
        self.trade_penalty = 0.005

    def reset(self):
        self.idx = 1

#        try:
#            self.day, self.stock, self.x = next(self.train_iter)
#        except StopIteration:
#            self.train_iter = iter(self.train_loader)
#            self.day, self.stock, self.x = next(self.train_iter)
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

        a = 2
        ema6 = np.zeros_like(p)
        ema12 = np.zeros_like(p)
        l1 = 6
        l2 = 12

        for t in range(len(p)):
            ema6[t] = p[t] * (a / (1 + l1))
            ema12[t] = p[t] * (a / (1 + l2))
            if t == 0:
                ema6[t] = p[0]
                ema12[t] = p[0]
            else:
                ema6[t] += ema6[t - 1] * (1 - a / (1 + l1))
                ema12[t] += ema12[t - 1] * (1 - a / (1 + l2))

        self.macd = ema6 - ema12
        self.ema6 = ema6
        self.ema12 = ema12


#        p = np.diff(np.log(p)) * 1.0e2
        self.x = torch.FloatTensor(self.macd)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        self.actions2 = list()

        state = self.x[:, :self.idx, :]

        state = torch.cat((state, torch.zeros_like(state)), dim=-1)

        self.hold_start = 0

        return state

    def step(self, action):
        self.actions.append(action)

        reward = 0.0

        self.actions2.append(action)

        p = self.prices[self.idx]

#        pprint(self.x[0, self.idx, 0])

#        if self.idx == 198:
#            print(f'{self.idx} {action}')

        if self.idx == 198:
            action = 0
#            print(f'clip: {self.idx} : {self.x.shape[1]}')

#        if self.idx > 190:
#            print(f'{self.idx} {action}')

        if action == 0:

            self.flat_count += 1

            # Sell point.
            if self.hold:
                self.hold = False
                self.sell_pts.append((self.idx, p))

                hold_dist = (self.idx - self.hold_start) * 1.0
#                if hold_dist < 4:
#                    hold_dist = -hold_dist

                hold_dist = (hold_dist**2) * 0.01

                reward = p - self.trade_penalty + hold_dist
                self.net += p - self.trade_penalty

        else:

            self.long_count += 1

            # Buy point.
            if not self.hold:
                self.hold = True
                self.buy_pts.append((self.idx, p))
                reward = -p - self.trade_penalty
                self.hold_start = self.idx
                self.net += -p - self.trade_penalty

        self.idx += 1
        state = self.x[:, :self.idx, :]

        done = self.idx + 1 >= self.x.shape[1]

        num_pts = len(self.buy_pts) + len(self.sell_pts)
        if done and num_pts < 4:
            reward -= 0.5

        q = torch.FloatTensor(self.actions2)
        q = torch.cat((torch.FloatTensor((0.0,)), q))
        q = q.reshape((1, q.shape[0], 1))
        state = torch.cat((state, q), dim=-1)
#        print(f'state: {state.shape}')
#        state = torch.cat((state, torch.zeros_like()), dim=-1)

        return state, reward, done
