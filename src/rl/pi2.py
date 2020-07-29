from pprint import pprint

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

import ticks_dataset


class Environment:
    def __init__(self):
        self.observe_n = 3
        self.action_n = 2

        self.mode = 'train'

        self.train_loader = DataLoader(ticks_dataset.TicksDataset(mode='train'),
                                       batch_size=1,
                                       shuffle=False)
        self.test_loader = DataLoader(ticks_dataset.TicksDataset(mode='test'),
                                      batch_size=1,
                                      shuffle=False)
        self.train_iter = iter(self.train_loader)

    def set_train(self):
        self.mode = 'train'

    def set_test(self):
        self.mode = 'test'

    def reset(self):
        self.tick_idx = 0
        self.day, self.symbol, self.x = next(self.train_iter)

#        self.tick_idx += 1
        return self.x[:, 0, :]

    def step(self, action):
        past_price = self.x[:, self.tick_idx, :]
        next_price = self.x[:, self.tick_idx + 1, :]

        if action.data[0] == 0:
            reward = 0.0
        elif action.data[0] == 1:
            reward = next_price - past_price

        self.tick_idx += 1
        done = self.tick_idx + 1 >= len(self)

        return self.x[:, self.tick_idx, :], reward, done

    def __len__(self):
        return self.x.shape[1]


class Policy(nn.Module):

    def __init__(self, env: Environment, device='cpu'):

        super(Policy, self).__init__()

        self.state_space = env.observe_n
        self.action_n = env.action_n
        self.gamma = 0.99

        self.lstm1 = nn.LSTM(input_size=self.state_space,
                             hidden_size=100,
                             num_layers=1,
                             bias=False,
                             batch_first=True)
        self.hidden_cell = (torch.zeros(1, 1, 100).to(device),
                            torch.zeros(1, 1, 100).to(device))

        self.fc1 = nn.Linear(in_features=100,
                             out_features=50)
        self.fc2 = nn.Linear(in_features=50,
                             out_features=self.action_n)

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):

        x = x.reshape((1, 1, 3))
        lstm_out, self.hidden_cell = self.lstm1(x, self.hidden_cell)
        x = self.fc1(lstm_out[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x


def select_action(state, policy):

    state = policy(Variable(state))
    c = torch.distributions.Categorical(state)
    action = c.sample()

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history,
                                           c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def main():
    env = Environment()
    policy = Policy(env)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    EPOCHS = 1

    for epoch in range(EPOCHS):

        state = env.reset()
        pprint(state)
        done = False

#        for tick_idx in range(len(env)):
        for tick_idx in range(5):
            action = select_action(state, policy)
            state, reward, done = env.step(action)

            policy.reward_episode.append(reward)
            if done:
                break

        update_policy(policy, optimizer)


def update_policy(policy: Policy, optimizer):
    rew = 0
    rewards = list()

    for r in policy.reward_episode[::-1]:
        rew = r + policy.gamma * rew
        rewards.insert(0, rew)
        print(rew)

    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    loss = (torch.sum(torch.mul(policy.policy_history,
                                Variable(rewards)).mul(-1), -1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = list()


if __name__ == '__main__':
    main()
