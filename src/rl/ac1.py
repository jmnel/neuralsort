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
from environment import Environment

GAMMA = 0.99
EPISODES = 100000


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        self.tcn = WaveNetNN(layers=6,
                             blocks=4,
                             dilation_channels=32,
                             residual_channels=32,
                             skip_channels=128,
                             end_channels=128,
                             input_channels=1,
                             output_channels=128,
                             classes=1,
                             output_length=1,
                             kernel_size=2)

        self.pad = nn.ConstantPad1d((0, 200), 0.0)

        self.value_head = nn.Linear(128, 1)
        self.action_head = nn.Linear(128, 2)

        self.saved_actions = list()
        self.rewards = list()

    def forward(self, x):

        x = x.reshape((1, 1, x.shape[1]))

        x = self.pad(x)
        x = x[:, :, -200:]

        x = self.tcn(x)

        action = F.softmax(self.action_head(x), dim=-1)
        value = self.value_head(x)

        return action, value


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):

    state = state[:, -200:, :]

    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    model.saved_actions.append((m.log_prob(action), state_value))

    return action.item()


def finish_episode():

    R = 0
    saved_actions = model.saved_actions
    policy_losses = list()
    value_losses = list()
    returns = list()

    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        policy_losses.append(-log_prob * advantage)

#        print(value.shape)
#        print(torch.tensor([R, ]).shape)
#        exit()
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).reshape_as(value)))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


env = Environment(trade_penalty=1e-3)


def main():

    running_reward = 10

    avg_rewards = list()
    last_rewards = list()
    episodes = list()

    for episode in range(EPISODES):

        state = env.reset()
        episode_reward = 0

        done = False
        while not done:

            action = select_action(state)

            state, reward, done = env.step(action)

            model.rewards.append(reward)
            episode_reward += reward

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        finish_episode()

        episodes.append(episode)
        avg_rewards.append(running_reward)
        last_rewards.append(episode_reward)

        fig, (ax1, ax2) = plt.subplots(2, 1)

        prices = env.prices
        ax1.plot(np.arange(len(prices)), prices, color='black', linewidth=0.2)

        ax1.plot(np.arange(len(prices)), env.ema6, color='red', linewidth=0.5)
        ax1.plot(np.arange(len(prices)), env.ema12, color='blue', linewidth=0.5)
        ax1.plot(np.arange(len(prices)), 5 * env.macd + np.mean(env.prices), color='magenta', linewidth=0.5)
        ax1.plot((0, len(prices) - 1), (np.mean(env.prices), np.mean(prices)), color='black', linewidth=0.2)

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

        plt.show()

        print('Episode {}, last reward: {}, avg. reward: {}'.format(
            episode, episode_reward, running_reward))
        print(f'Buy: {len(env.buy_pts)}, sell: {len(env.sell_pts)}')
        print(f'Net: {env.net}')


if __name__ == '__main__':
    main()
