from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

import numpy as np

from tcn.wavenet_nn import WaveNetNN
from ticks_dataset import TicksDataset, TicksDatasetIEX

MAX_EPISODES = 100000
GAMMA = 0.99
N_TRIALS = 1
REWARD_THRESHOLD = 400
HIDDEN_SIZE = 50
N_STEP = 10
DEVICE = 'cuda'
NUM_LAYERS = 1
PENALTY = 0.0005

run_hist = list()


class Environment:

    def __init__(self):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

    def reset(self):
        self.idx = 0
        self.buy_price = 0.0
        self.is_long = False
        self.actions = list()
        self.flat_count = 0
        self.long_count = 0

        self.cash = 1.0
        self.hold = 0.0

        self.cash_hist = [self.cash, ]
        self.hold_hist = [self.hold, ]

        self.buy_pts = list()
        self.sell_pts = list()

        self.net = 0

        q = iter(self.train_loader)
        self.day, self.stock, self.x = next(q)

        self.prices = self.x

        p = self.x[0, :, 1].numpy()
        p = p / p[0]

        p = p[:200]
        self.prices = p
        p = np.diff(np.log(p)) * 1.0e2
        self.x = torch.FloatTensor(p)
        self.x = self.x.reshape((1, self.x.shape[0], 1))

        state = self.x[:, :self.idx + 1, :]

        return state

    def step(self, action):
        self.idx += 1
        state = self.x[:, :self.idx + 1, :]

        r = self.x[0, self.idx, 0]

        done = False

        max_sale = (-float('inf'), 0, 0)
        # Action is flat = 0.
        if action == 0:
            self.flat_count += 1
#            reward = -r
            if self.hold > 0.0:
                self.hold = 0.0
                reward = self.prices[self.idx] - self.buy_price - PENALTY
#                print(f'sell @{self.idx+1} R={reward}')
                self.sell_pts.append(self.idx)
                self.net += reward
            else:
                reward = 0.0

        # Action is long = 1.
        elif action == 1:
            self.long_count += 1

            if self.hold <= 0.0:
                self.hold = 1.0
                self.buy_pts.append(self.idx)
                self.buy_price = self.prices[self.idx]
                reward = -self.buy_price - PENALTY
                self.net += -self.buy_price - PENALTY
#                print(f'buy @{self.idx+1} R={reward}')
            else:
                reward = 0.0

        done = self.idx + 1 >= self.x.shape[1]

#        print(max_sale)

        self.actions.append(action)

        return state, reward, done


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.tcn = WaveNetNN(layers=6,
                             blocks=4,
                             dilation_channels=32,
                             residual_channels=32,
                             skip_channels=128,
                             end_channels=128,
                             input_channels=1,
                             output_channels=2,
                             classes=1,
                             output_length=1,
                             kernel_size=2)

        self.pad = nn.ConstantPad1d((0, 256), 0.0)

    def forward(self, x):

        x = x.reshape((1, 1, x.shape[1]))
        x = self.pad(x)
        x = x[:, :, :256]
        y = self.tcn(x)
        return y


policy = Policy().to(DEVICE)
train_env = Environment()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

actions = list()


def train(env, policy, optimizer, gamma):

    policy.train()

    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()
    global actions
    actions = list()

    while not done:

        state = state.to(DEVICE)

        state = state[:, -200:, :]

        action_pred = policy(state)

        action_prob = F.softmax(action_pred, dim=1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()
        log_prob_action = dist.log_prob(action).reshape((1, 1))

        state, reward, done = env.step(action.item())

        actions.append(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions, dim=0)

    flat_pts = list()
    long_pts = list()

    returns = calculate_returns(rewards, gamma)

    loss = update_policy(returns.to(DEVICE), log_prob_actions.to(
        DEVICE), optimizer)

    return loss, episode_reward


def calculate_returns(rewards, gamma, normalize=True):

    returns = list()
    R = 0

    for r in reversed(rewards):
        R = r + R * gamma
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def update_policy(returns, log_prob_actions, optimizer):

    returns = returns.detach()
#    run = run.detach()
    loss = -(returns * log_prob_actions).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(env, policy):

    policy.eval()

    done = False
    episode_reward = 0

    state = env.reset()

#    while not


train_rewards = list()
test_rewards = list()

reward_hist = list()
mv_hist = list()

for episode in range(1, MAX_EPISODES + 1):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)

    loss, train_reward = train(train_env, policy, optimizer, GAMMA)

    run_hist.append((episode, train_env.idx))

    train_rewards.append(train_reward)

    avg_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mv_hist.append(np.mean(train_rewards[-50:]))

#    max_raw_hist.append(np.max(train_rewards[-N_TRIALS:]))
#    min_raw_hist.append(np.min(train_rewards[-N_TRIALS:]))

    reward_hist.append(avg_train_rewards)

    x = train_env.prices
    ax1.plot(np.arange(len(x)), x, color='black', linewidth=0.2)
    for i, action in enumerate(actions):
        if action == 0:
            color = 'red'
        else:
            color = 'green'

#        ax1.plot(tuple(range(i + 0, i + 2)), x[0, i + 0:i + 2, 1], color=color, linewidth=0.4)
        ax1.plot(tuple(range(i + 1, i + 3)), x[i + 1:i + 3], color=color, linewidth=0.4)

    ax1.scatter(tuple(t for t in train_env.buy_pts), tuple(x[t] for t in train_env.buy_pts), color='green', s=8)
    ax1.scatter(tuple(t for t in train_env.sell_pts), tuple(x[t] for t in train_env.sell_pts), color='red', s=8)

    ax2.plot(tuple(range(len(reward_hist))), reward_hist, linewidth=0.4)
    ax2.plot(tuple(range(len(reward_hist))), mv_hist, linewidth=0.4)
#    ax2.plot(tuple(range(len(reward_hist))), max_raw_hist, linewidth=0.4, color='C3')
#    ax2.plot(tuple(range(len(reward_hist))), min_raw_hist, linewidth=0.4, color='C4')
    plt.show()

    print(f'Episode: {episode}, avg. train reward: {avg_train_rewards}, loss: {loss}')
    print(f'Buy: {len(train_env.buy_pts)}, sell: {len(train_env.sell_pts)}')
    print(f'Flat: {train_env.flat_count}, long: {train_env.long_count}')
    print(f'Net: {train_env.net}')
