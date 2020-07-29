import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Environment:
    def __init__(self):
        self.observe_n = 3
        self.action_n = 2

        self.mode = 'train'

    def set_train(self):
        self.mode = 'train'

    def set_test(self):
        self.mode = 'test'


class PolicyNN(nn.Module):

    def __init__(self, env):
        super(PolicyNN, self).__init__()

        self.state_space = env.observe_n
        self.action_space = env.action_n

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = 0.99

        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        y = self.l1(x)
        y = nn.Dropout(p=0.6)
        y = nn.ReLU()
        y = self.l2(y)
        y = nn.Softmax(dim=-1)


policy = PolicyNN()
optimizer = optim.Adam(policy.parameters(), lr=1 - 2)


def select_action(state):

    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = torch.distributions.Categorical(state)
    action = c.sample()

    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history,
                                           c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def update_policy():
    rew = 0
    rewards = []

    # Discount
    for r in policy.reward_episode[::-1]:
        rew = r + policy.gamma * rew
        rewards.insert(0, rew)

    rewards = torch.FloatType(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    loss = (torch.sum(torch.mul(policy.policy_history,
                                Variable(rewards)).mul(-1), -1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def main(episodes):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        done = False

        for time in range(1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action.data[0])

            policty.reward_episode.append(reward)
            if done:
                break

        running_reward = (running_reward * 0.99) + (time * 0.01)
