import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 512


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):

    def __init__(self, state_space):
        super(Critic, self).__init__()

        self.state_space = state_space
        self.linear1 = nn.Linear(state_space, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear6 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear7 = nn.Linear(HIDDEN_SIZE, 2)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
        self.elu5 = nn.ELU()
        self.elu6 = nn.ELU()

        self.apply(weights_init_)

    def forward(self, state):

        x = self.linear1(state)
        x = self.linear2(self.elu1(x))
        x = self.linear3(self.elu2(x))
        x = self.linear4(self.elu3(x))
        x = self.linear5(self.elu4(x))
        x = self.linear6(self.elu5(x))
        x = self.linear7(x)

        return x


class Actor(nn.Module):

    def __init__(self, state_space):
        super(Actor, self).__init__()

        self.state_space = state_space

        self.linear1 = nn.Linear(state_space, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear6 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear7 = nn.Linear(HIDDEN_SIZE, 2)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
        self.elu5 = nn.ELU()

        self.apply(weights_init_)

    def forward(self, state):

        x = self.linear1(state)
        x = self.linear2(self.elu1(x))
        x = self.linear3(self.elu2(x))
        x = self.linear4(self.elu3(x))
        x = self.linear5(self.elu4(x))
        x = self.linear6(self.elu5(x))
        x = self.linear7(x)

        return F.softmax(x, dim=-1)
