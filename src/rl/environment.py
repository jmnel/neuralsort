from pprint import pprint

import torch
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

import numpy as np

from ticks_dataset import TicksDataset, TicksDatasetIEX


class Environment:

    def __init__(self):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

        self.idx = 0
        self.init_len = 3

    def reset(self):
        self.idx = self.init_len
        self.day, self.stock, self.x = next(iter(self.train_loader))

        self.actions = list()
        self.buy_pts = list()
        self.sell_pts = list()

        p = self.x[0, :, 1].numpy()
        p = p[:200]
        self.prices = p

        p = p / p[0]
        p = np.diff(np.log(p)) * 1.0e2
        self.x = torch.FloatTensor(p)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        state = self.x[:, :self.idx, :]

        return state

    def step(self, action):
        self.idx += 1
        state = self.x[:, :self.idx, :]

        done = self.idx >= self.x.shape[1]

        reward = 0.0

        return state, reward, done


env = Environment()

state = env.reset()
print(state.shape)


done = False
print(env.x.shape[1])
while not done:

    print(env.idx)

    p1 = env.prices[env.idx]
    p2 = env.prices[env.idx + 1]

    r = p2 - p1
    if r > 0.0:
        action = 1
    else:
        action = 0

    state, reward, done = env.step(action)

    actions.append(action)

prices = env.prices
t1 = np.arange(len(prices))

x = env.x[0, :, 0]
t2 = np.arange(1, len(x) + 1)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t1, prices, linewidth=0.2, color='black')

for idx, a in enumerate(actions):
    i = idx + 3
    if a == 0:
        c = 'red'
    else:
        c = 'green'

    ax1.plot(t1[i:i + 2], prices[i: i + 2], color=c, linewidth=0.4)

ax2.plot(t2, x, linewidth=0.4, color='C1')
plt.show()
