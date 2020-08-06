import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(10, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2)

#        self.elu1 = nn.ELU()
#        self.elu2 = nn.ELU()

    def forward(self, state):

        x = self.linear1(state)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))

        return x


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(10, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 2)

#        self.elu1 = nn.ELU()
#        self.elu2 = nn.ELU()

    def forward(self, state):

        #        print(f'state: {state}')

        x = self.linear1(state)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))

        return F.softmax(x, dim=-1)
