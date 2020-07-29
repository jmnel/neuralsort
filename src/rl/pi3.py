from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from ticks_dataset import TicksDataset

MAX_EPISODES = 250
GAMMA = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 400
LONG_PENALTY = 0.01
HIDDEN_SIZE = 10
N_STEP = 100
DEVICE = 'cuda'
NUM_LAYERS = 3


class Environment:

    def __init__(self):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

    def reset(self):
        self.idx = 0

        try:
            self.day, self.stock, self.x = next(self.train_iter)
#            self.day, self.stock, self.x = next(iter(self.train_loader))

        except StopIteration:
            self.train_iter = iter(self.train_loader)
            self.day, self.stock, self.x = next(self.train_iter)


#        pprint(self.x[0, :, 0])
#        plt.plot(self.x[0, :, 0], self.x[0, :, 1], linewidth=0.4)
#        plt.show()

        state = self.x[0, :self.idx + 1, :]

        return state

    def step(self, action):
        self.idx += 1
        state = self.x[0, :self.idx + 1, :]

        # Action is flat = 0.
        if action == 0:
            reward = 0.0

        # Action is long = 1
        elif action == 1:
            reward = self.x[0, self.idx, 1] - self.x[0, self.idx - 1, 1] - LONG_PENALTY

        done = self.idx + 1 >= self.x.shape[1]

        return state, reward, done


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(3, HIDDEN_SIZE, batch_first=True, dropout=0.5, num_layers=NUM_LAYERS)

        self.fc1 = nn.Linear(HIDDEN_SIZE, 128)
        self.fc2 = nn.Linear(128, 2)
        self.hidden_cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                            torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm1(x, self.hidden_cell)
        x = F.relu(lstm_out[:, -1, :])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


policy = Policy().to(DEVICE)
train_env = Environment()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def train(env, policy, optimizer, gamma):

    policy.train()

    log_prob_actions = []
    rewards = []
    done = False
    episode_reward = 0

    state = env.reset()

    plt.ion()
    actions = list()
    while not done:

        state = torch.FloatTensor(state).reshape((1, state.shape[0], 3)).to(DEVICE)

#        action_pred = policy(state[:, 1, :])
        action_pred = policy(state[:, -N_STEP:, :])
        action_prob = F.softmax(action_pred, dim=-1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, done = env.step(action[0].item())

        actions.append(action[0].item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

    log_prob_actions = torch.cat(log_prob_actions)

    plt.clf()
    plt.plot(state[:, 0], state[:, 1], linewidth=0.4)

    flat_pts = list()
    long_pts = list()

    for idx, action in enumerate(actions):
        if action == 0:
            plt.scatter(state[idx:idx + 1, 0], state[idx:idx + 1, 1], color='r', s=4)
        elif action == 1:
            plt.scatter(state[idx:idx + 1, 0], state[idx:idx + 1, 1], color='g', s=4)

    plt.pause(0.01)

    returns = calculate_returns(rewards, gamma)

    loss = update_policy(returns.to(DEVICE), log_prob_actions.to(DEVICE), optimizer)

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


for episode in range(1, MAX_EPISODES + 1):

    policy.hidden_cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                          torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

    loss, train_reward = train(train_env, policy, optimizer, GAMMA)

    train_rewards.append(train_reward)

    avg_train_rewards = np.mean(train_rewards[-N_TRIALS:])

    print(f'Episode: {episode}, avg. train reward: {avg_train_rewards}')
