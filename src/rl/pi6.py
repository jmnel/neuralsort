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
from environment2 import Environment

MAX_EPISODES = 100000
GAMMA = 0.99
N_TRIALS = 1
REWARD_THRESHOLD = 400
HIDDEN_SIZE = 50
N_STEP = 10
DEVICE = 'cuda'
NUM_LAYERS = 1
PENALTY = 0.005
INVALID_PENALTY = 0.1

run_hist = list()


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.tcn = WaveNetNN(layers=6,
                             blocks=4,
                             dilation_channels=32,
                             residual_channels=32,
                             skip_channels=128,
                             end_channels=128,
                             input_channels=3,
                             output_channels=3,
                             classes=1,
                             output_length=1,
                             kernel_size=2)

        self.pad = nn.ConstantPad1d((0, 256), 0.0)

#        self.fc1 = nn.Linear(3 + 1, 12)
#        self.fc2 = nn.Linear(12, 3)

    def forward(self, state):

        x = state

        x = x.reshape((1, 3, x.shape[1]))
        x = self.pad(x)
        x = x[:, :, :256]
        x = self.tcn(x)

        return x

#        z = torch.cat((y, hold_f), dim=1)
#        z = self.fc1(z)
#        z = F.relu(z)

#        z = self.fc2(z)
#        y += torch.autograd.Variable(torch.randn(y.size()).to(DEVICE)) * 0.1
#        return z


policy = Policy().to(DEVICE)
train_env = Environment(PENALTY, INVALID_PENALTY)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# optimizer = optim.SGD(policy.parameters(), lr=1e-3)

actions = list()

runtime = 0.0


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

    global runtime
    runtime = (train_env.x.shape[1] - train_env.idx) * 0.1

    loss = update_policy(returns.to(DEVICE), log_prob_actions.to(
        DEVICE), optimizer) - runtime

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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, sharey=False)

    loss, train_reward = train(train_env, policy, optimizer, GAMMA)

    run_hist.append((episode, train_env.idx))

    train_rewards.append(train_reward)

    avg_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mv_hist.append(np.mean(train_rewards[-50:]))

#    max_raw_hist.append(np.max(train_rewards[-N_TRIALS:]))
#    min_raw_hist.append(np.min(train_rewards[-N_TRIALS:]))

    reward_hist.append(avg_train_rewards)

    prices = train_env.prices
    ax1.plot(np.arange(len(prices)), prices, color='black', linewidth=0.2)
#    for idx, action in enumerate(train_env.actions):
#        i = idx + 3
#        if action == 0:
#            color = 'red'
#        else:
#            color = 'green'

#        ax1.plot(tuple(range(i, i + 2)), prices[i: i + 2], color=color, linewidth=0.4)

    if len(train_env.buy_pts) > 0:
        ax1.scatter(*(zip(*train_env.buy_pts)), color='green', s=8)
    if len(train_env.sell_pts) > 0:
        ax1.scatter(*(zip(*train_env.sell_pts)), color='red', s=8)

    ax2.plot(tuple(range(len(reward_hist))), reward_hist, linewidth=0.4)
    ax2.plot(tuple(range(len(reward_hist))), mv_hist, linewidth=0.6)

    ax3.plot(tuple(range(min(200, len(reward_hist)))), reward_hist[-200:], linewidth=0.4)
    ax3.plot(tuple(range(min(200, len(reward_hist)))), mv_hist[-200:], linewidth=0.6)
#    ax2.plot(tuple(range(len(reward_hist))), max_raw_hist, linewidth=0.4, color='C3')
#    ax2.plot(tuple(range(len(reward_hist))), min_raw_hist, linewidth=0.4, color='C4')
    plt.show()

    print(f'Episode: {episode}, avg. train reward: {avg_train_rewards}, loss: {loss}')
    print(f'Buy: {len(train_env.buy_pts)}, sell: {len(train_env.sell_pts)}, hold: {train_env.hold_count}')
    print(f'invalid: {train_env.invalid_count}')
#    print(f'Flat: {train_env.flat_count}, long: {train_env.long_count}')
    print(f'Net: {train_env.net}')
    print(f'sell_r: {train_env.sell_raw}, buy_r: {train_env.buy_raw}, hold_r: {train_env.hold_raw}')
    print(f'runtime: {runtime}')
#    p_reward_avg = np.mean(train_env.p_rewards)
#    r_reward_avg = np.mean(train_env.r_rewards)
#    print(f'p reward: {p_reward_avg}, r reward: {r_reward_avg}')
#    print(f'hold len: {train_env.max_hold_len}')
