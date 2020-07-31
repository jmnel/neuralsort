from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.utils.data import DataLoader

from tcn.wavenet_nn import WaveNetNN

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Cairo')
import numpy as np

from ticks_dataset import TicksDataset, TicksDatasetIEX

MAX_EPISODES = 1000
GAMMA = 0.99
N_TRIALS = 1
REWARD_THRESHOLD = 400
LONG_PENALTY = 0.01
HIDDEN_SIZE = 50
N_STEP = 10
DEVICE = 'cpu'
NUM_LAYERS = 1
PENALTY = 0.01

run_hist = list()


class Environment:

    def __init__(self):
        self.train_loader = DataLoader(TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.train_iter = iter(self.train_loader)

    def reset(self):
        self.net = 0.0
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

        q = iter(self.train_loader)
        self.day, self.stock, self.x = next(q)

        self.prices = self.x

        p = self.x[0, :, 1].numpy()
        p = p / p[0]
#        p = np.diff(np.log(p)) * 1.0e3
        self.x = torch.FloatTensor(p)
#        pprint(self.x)
        self.x = self.x.reshape((1, self.x.shape[0], 1))


#        print(self.stock)
#        try:
#            self.day, self.stock, self.x = next(self.train_iter)

#        except StopIteration:
#            self.train_iter = iter(self.train_loader)
#            self.day, self.stock, self.x = next(self.train_iter)


#        pprint(self.x[0, :, 0])
#        plt.plot(self.x[0, :, 0], self.x[0, :, 1], linewidth=0.4)
#        plt.show()

#        ch = torch.FloatTensor(self.cash_hist).reshape(1, 1)
#        hh = torch.FloatTensor(self.hold_hist).reshape(1, 1)

        state = self.x[:, :self.idx + 1, :]
#        state = torch.cat((self.x[0, :self.idx + 1, :],
#                           ch, hh), dim=1)

        return state

    def step(self, action):
        self.idx += 1
        state = self.x[:, :self.idx + 1, :]

        new_price = self.x[0, self.idx, 0]
        prev_price = self.x[0, self.idx - 1, 0]

        done = False

        # Action is flat = 0.
        if action == 0:
            self.flat_count += 1
#            reward = 0
            if self.hold > 0.0:
                #                reward -= 0.005
                self.hold = 0.0
                reward = new_price - self.buy_price - 0.005
                self.sell_pts.append(self.idx)

                #            if self.hold > 0.0:
                #                self.cash = self.hold * new_price - 0.005
                #                reward = self.cash
                #                self.hold = 0.0
                #                self.sell_pts.append(self.idx - 1)
                #            else:
                #                reward = self.cash

                # Action is long = 1.
            else:
                reward = 0.0
        elif action == 1:
            self.long_count += 1
            reward = new_price - prev_price

#            reward = new_price
            if self.hold <= 0.0:
                reward = -new_price - 0.005
                self.buy_price = new_price
                self.hold = 1.0
                self.buy_pts.append(self.idx)
            else:
                reward = 0.0

#            if self.hold <= 0.0:

            #                self.hold = self.cash / new_price - 0.005
            #                self.cash = 0.0
            #                reward = self.hold * new_price
            #                self.buy_pts.append(self.idx - 1)

            #                reward = new_price - prev_price - LONG_PENALTY
            #                reward = -1.0 - PENALTY
            #                self.net -= new_price
            #                self.is_long = True
            #                self.buy_price = new_price
#            else:
            #                reward = new_price * self.hold
            #                reward = new_price - prev_price
            #            reward = self.x[0, self.idx, 1] - self.x[0, self.idx - 1, 1]
            #            if len(self.actions) == 0:
            #                reward -= LONG_PENALTY
            #            elif self.actions[-1] != 1:
            #                reward -= LONG_PENALTY

        done = self.idx + 1 >= self.x.shape[1]

        self.actions.append(action)

#        self.cash_hist.append(self.cash)
#        self.hold_hist.append(self.hold)

#        ch = torch.FloatTensor(self.cash_hist).reshape(len(self.hold_hist), 1)
#        hh = torch.FloatTensor(self.hold_hist).reshape(len(self.hold_hist), 1)

#        state = torch.cat((state,
#                           ch,
#                           hh), dim=1)

        return state, reward, done


class Policy(nn.Module):

    def __init__(self):
        super().__init__()

#        self.lstm1 = nn.LSTM(3, HIDDEN_SIZE, batch_first=True, dropout=0.5, num_layers=NUM_LAYERS)
#        self.lstm1 = nn.LSTM(1, HIDDEN_SIZE, batch_first=True, num_layers=NUM_LAYERS)

        self.tcn = WaveNetNN(layers=10,
                             blocks=4,
                             dilation_channels=32,
                             residual_channels=32,
                             skip_channels=256,
                             end_channels=256,
                             input_channels=1,
                             output_channels=2,
                             classes=1,
                             output_length=1,
                             kernel_size=2)

#        self.fc1 = nn.Linear(HIDDEN_SIZE, 128)
#        self.fc1 = nn.Linear(6048, 1024)
#        self.fc2 = nn.Linear(128, 2)
#        self.hidden_cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
#                            torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

#        self.conv1 = nn.Conv1d(1, 8, 1)
#        self.conv2 = nn.Conv1d(8, 16, 2)
#        self.conv3 = nn.Conv1d(16, 32, 4)
#        self.conv4 = nn.Conv1d(32, 32, 8)

        self.pad = nn.ConstantPad1d((0, 4000), 0.0)

    def forward(self, x):

        x = x.reshape((1, 1, x.shape[1]))
#        print(x.shape)
        x = self.pad(x)
#        print(x.shape)
#        x = x[:,
        y = self.tcn(x)
        return y

        #        x = x.reshape((1, 1, x.shape[1]))

        #        x = self.pad(x)
        #        x = x[:, :, 0:200]

        #        x = self.conv1(x)
        #        x = F.relu(x)

        #        x = self.conv2(x)
        #        x = F.relu(x)

        #        x = self.conv3(x)
        #        x = F.relu(x)

        #        x = self.conv4(x)
        #        x = F.relu(x)

        #        x = x.flatten()

        #        x = self.fc1(x)
        #        x = self.fc2(x)

        #        x = F.softmax(x, dim=-1)

        #        return x

        #        exit()

#        lstm_out, self.hidden_cell = self.lstm1(x, self.hidden_cell)
#        x = F.relu(lstm_out[:, -1, :])
#        x = self.fc1(x)
#        x = F.relu(x)
#        x = self.fc2(x)
#        x = F.softmax(x, dim=-1)
        return x


policy = Policy().to(DEVICE)
train_env = Environment()
optimizer = optim.Adam(policy.parameters(), lr=1e-2, weight_decay=1e-5)

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

        #        state = torch.FloatTensor(state).reshape((1, state.shape[0], 3)).to(DEVICE)
        state = state.to(DEVICE)

#        action_pred = policy(state[:, 1, :])
        action_pred = policy(state[:, -N_STEP:, :])

#        print(action_pred)
        action_prob = F.softmax(action_pred, dim=1)
#        print(action_prob)

        dist = distributions.Categorical(action_prob)

#        print(dist)
        action = dist.sample()
        log_prob_action = dist.log_prob(action).reshape((1, 1))

#        print(log_prob_action)

        state, reward, done = env.step(action.item())

        actions.append(action.item())

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)

        episode_reward += reward

#    print(log_prob_actions)
#    print(log_prob_actions.shape)
#    pprint(log_prob_actions)
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

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)
for episode in range(1, MAX_EPISODES + 1):

    policy.hidden_cell = (torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                          torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

    ax1.clear()
    ax2.clear()

    loss, train_reward = train(train_env, policy, optimizer, GAMMA)

    run_hist.append((episode, train_env.idx))

    train_rewards.append(train_reward)

    avg_train_rewards = np.mean(train_rewards[-N_TRIALS:])

    reward_hist.append(avg_train_rewards)

    x = train_env.prices
    for i, action in enumerate(actions):
        if action == 0:
            color = 'red'
        else:
            color = 'green'

        ax1.plot(tuple(range(i + 0, i + 2)), x[0, i + 0:i + 2, 1], color=color, linewidth=0.4)
    ax2.plot(tuple(range(len(reward_hist))), reward_hist, linewidth=0.4)
    plt.pause(0.01)

    print(f'Episode: {episode}, avg. train reward: {avg_train_rewards}, loss: {loss}')
    print(f'Buy: {len(train_env.buy_pts)}, sell: {len(train_env.sell_pts)}')
    print(f'Flat: {train_env.flat_count}, long: {train_env.long_count}')
    print(f'Net: {train_env.net}\n')
