import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt

from environment4 import Environment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n
lr = 1e-2
TAU = 2.0e-5

HIDDEN_SIZE = 100
LSTM_LAYERS = 1
LSTM_FEATURES = 1


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.lstm = nn.LSTM(input_size=LSTM_FEATURES,
                            hidden_size=HIDDEN_SIZE,
                            batch_first=True,
                            bias=False,
                            num_layers=LSTM_LAYERS)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

        self.fc1 = nn.Linear(HIDDEN_SIZE, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256 + 1, 2)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

    def forward(self, state):

        prices, prev_action = state

        output, self.hidden_cell = self.lstm(prices, self.hidden_cell)
        output = output[:, -1, :]
        output = self.elu1(self.fc1(output))
        output = self.elu2(self.fc2(output))

        output = self.fc3(torch.cat((output, prev_action), dim=-1))

        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.lstm = nn.LSTM(input_size=LSTM_FEATURES,
                            hidden_size=HIDDEN_SIZE,
                            batch_first=True,
                            bias=False,
                            num_layers=LSTM_LAYERS)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

        self.fc1 = nn.Linear(HIDDEN_SIZE, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256 + 1, 1)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

    def forward(self, state):
        prices, prev_action = state

        output, self.hidden_cell = self.lstm(prices, self.hidden_cell)
        output = output[:, -1, :]
        output = self.elu1(self.fc1(output))
        output = self.elu2(self.fc2(output))
        value = self.fc3(torch.cat((output, prev_action), dim=-1))
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


env = Environment(trade_penalty=0.05)
#values_hist = list()
reward_hist = list()
net_hist = list()
running_reward = -12
running_net = 0
#running_value = 0
running_reward_hist = list()
running_net_hist = list()
#running_value_hist = list()


def trainIters(actor, critic, n_iters):
    optimizer_a = optim.Adam(actor.parameters())
    optimizer_c = optim.Adam(critic.parameters())
    for epsiode in range(1000000):

        actor.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                             torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))
        critic.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE),
                              torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE))

        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        episode_reward = 0

        for i in count():
            prices, prev_action = state
            prices = prices.to(DEVICE)
            prev_action = prev_action.to(DEVICE)
            prev_action = prev_action.reshape((1, 1))

#            = torch.FloatTensor(state).to(DEVICE)
            dist = actor((prices, prev_action))
            value = critic((prices, prev_action))

            action = dist.sample()

#            print(action)

            next_state, reward, done = env.step(action.item())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=DEVICE))
            masks.append(torch.tensor([1 - done], dtype=torch.float, device=DEVICE))

            episode_reward += reward

            state = next_state

            if done:
                break

        reward_hist.append(episode_reward)
        net_hist.append(env.net)
        global running_reward
        global running_net
        if epsiode == 0:
            running_reward = episode_reward
            running_net = env.net
        running_reward = (1 - 0.05) * running_reward + 0.05 * episode_reward
        running_net = (1 - 0.01) * running_net + 0.01 * env.net
        running_reward_hist.append(running_reward)
        running_net_hist.append(running_net)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(np.arange(len(env.prices)), env.prices, linewidth=0.2, color='black')
        ax1.plot(np.arange(len(env.prices)), env.ema1, linewidth=0.2, color='blue')

        if len(env.buy_pts) > 0:
            ax1.scatter(*zip(*env.buy_pts), s=4, color='green')
        if len(env.sell_pts) > 0:
            ax1.scatter(*zip(*env.sell_pts), s=4, color='red')

        t = np.arange(epsiode + 1)

        ax2.plot(t, reward_hist, linewidth=0.2)
        ax2.plot(t, running_reward_hist, linewidth=0.6)

        ax3.plot(t, net_hist, linewidth=0.2)
        ax3.plot(t, running_net_hist, linewidth=0.6)

#        ax3.plot(t, values_hist, linewidth=0.3)
#        ax3.plot(t, running_value_hist, linewidth=0.6)

        plt.show()

        next_state = (torch.FloatTensor(next_state[0]).to(DEVICE),
                      torch.FloatTensor(next_state[1]).to(DEVICE).reshape((1, 1)))
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() - TAU * entropy
        critic_loss = advantage.pow(2).mean()

        print('Iteration: {}, reward: {}'.format(epsiode, episode_reward))
        print(f'Entropy: {entropy}')
        print('Loss -> actor: {:.4f}, critic: {:.4f}'.format(actor_loss, critic_loss))
        print(f'Buy pts. {len(env.buy_pts)}, sell pts. {len(env.sell_pts)}')

        optimizer_a.zero_grad()
        optimizer_c.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizer_a.step()
        optimizer_c.step()
#    torch.save(actor, 'model/actor.pkl')
#    torch.save(critic, 'model/critic.pkl')
#    env.close()


def main():
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    trainIters(actor, critic, n_iters=100)


if __name__ == '__main__':
    main()
