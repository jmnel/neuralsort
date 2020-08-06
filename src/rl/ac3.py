import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

from tcn.wavenet_nn import WaveNetNN
from environment3 import Environment

GAMMA = 0.9999
EPISODES = 100000
TRADE_PENALTY = 0.005
DEVICE = 'cuda'
HIDDEN_SIZE = 100
LSTM_LAYERS = 3
LSTM_DROPOUT = 0.5
#


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.lstm = nn.LSTM(input_size=3,
                            hidden_size=HIDDEN_SIZE,
                            batch_first=True,
                            bias=False,
                            num_layers=LSTM_LAYERS)
        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

        self.fc1 = nn.Linear(HIDDEN_SIZE + 2, 128)

        self.value_head = nn.Linear(128, 1)
        self.action_head = nn.Linear(128, 3)

        self.saved_actions = list()
        self.rewards = list()

        self.prelu = nn.PReLU()

    def forward(self, x):

        z = x[:, -1, 1:]

        x, self.hidden_cell = self.lstm(x, self.hidden_cell)

        x = x[:, -1, :]

        x = torch.cat((z, x), dim=-1)

        x = self.fc1(x)
        x = torch.tanh(x)

        action = self.action_head(x)
        action = F.softmax(action, dim=-1)

        value = self.value_head(x)

        return action, value


model = Policy().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):

    state = state[:, -200:, :].to(DEVICE)

    model.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                         torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

    probs, state_value = model(state)

#    print(f'probs: {probs.shape}')

    m = Categorical(probs)

    action = m.sample()

    model.saved_actions.append((m.log_prob(action), state_value))

#    print(action.shape)

    return action.item()


def finish_episode():

    R = 0
    saved_actions = model.saved_actions
    policy_losses = list()
    value_losses = list()
    returns = list()

    print(len(model.saved_actions))

    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value.to(DEVICE), torch.tensor([R]).reshape_as(value).to(DEVICE)))

    optimizer.zero_grad()

    policy_losses = torch.stack(policy_losses).to(DEVICE).sum()
    value_losses = torch.stack(value_losses).to(DEVICE).sum()

    loss = policy_losses + value_losses

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


env = Environment(trade_penalty=TRADE_PENALTY)


def main():

    running_reward = 0
    running_value = 0.6

    avg_rewards = list()
    last_rewards = list()
    episodes = list()

    values = list()
    running_values = list()

    for episode in range(EPISODES):

        state = env.reset()

#        print(state.shape)
        episode_reward = 0

        done = False
        while not done:

            action = select_action(state)

            state, reward, done = env.step(action)

            model.rewards.append(reward)
            episode_reward += reward

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        value = env.cash + (env.hold * env.prices[-1] - env.trade_penalty)
        running_value = 0.05 * value + (1 - 0.05) * running_value
        running_values.append(running_value)

        finish_episode()

        episodes.append(episode)
        avg_rewards.append(running_reward)
        last_rewards.append(episode_reward)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

        prices = env.prices
        ax1.plot(np.arange(len(prices)), prices, color='black', linewidth=0.2)

        for t0, t1 in env.flat_intervals:
            ax1.plot(tuple(range(t0, t1 + 1)), env.prices[t0:t1 + 1], color='red', linewidth=0.4)
        for t0, t1 in env.hold_intervals:
            ax1.plot(tuple(range(t0, t1 + 1)), env.prices[t0:t1 + 1], color='green', linewidth=0.4)
#        ax1.plot(np.arange(len(prices)), env.ema6, color='red', linewidth=0.5)
#        ax1.plot(np.arange(len(prices)), env.ema12, color='blue', linewidth=0.5)
#        ax1.plot(np.arange(len(prices)), 5 * env.macd + np.mean(env.prices), color='magenta', linewidth=0.5)
#        ax1.plot((0, len(prices) - 1), (np.mean(env.prices), np.mean(prices)), color='black', linewidth=0.2)

        for idx, action in enumerate(env.actions):
            i = idx + 0
            if action == 0:
                color = 'red'
            else:
                color = 'green'

#            ax1.plot(tuple(range(i, i + 2)), prices[i:i + 2], color=color, linewidth=0.4)

        if len(env.buy_pts) > 0:
            ax1.scatter(*(zip(*env.buy_pts)), color='green', s=8)
        if len(env.sell_pts) > 0:
            ax1.scatter(*(zip(*env.sell_pts)), color='red', s=8)

        ax2.plot(episodes, last_rewards, linewidth=0.4)
        ax2.plot(episodes, avg_rewards, linewidth=0.5)

        ax3.plot(episodes[-100:], last_rewards[-100:], linewidth=0.4)
        ax3.plot(episodes[-100:], avg_rewards[-100:], linewidth=0.4)

        values.append(value)
        ax4.plot(episodes, values, linewidth=0.3)
        ax4.plot(episodes, running_values, linewidth=0.5)
#        ax4.plot(episodes, env.cash_hist, linewidth=0.4)
#        ax4.plot(episodes, env.hold_hist, linewidth=0.4)

        plt.show()

        print('Episode {}, last reward: {}, avg. reward: {}'.format(
            episode, episode_reward, running_reward))
        print(f'Buy: {len(env.buy_pts)}, sell: {len(env.sell_pts)}')
#        print(f'Cash: {env.cash}, hold: {env.hold}, value: {value}, avg. value: {running_value}')
        print('Cash: {:.6f}, hold: {:.6f}, value: {:.6f}, avg. value: {:.6f}'.format(
            env.cash, env.hold, value, running_value))
        print(f'Invalid: {env.invalid_count}')
#        print(f'Net: {env.net}')


if __name__ == '__main__':
    main()
