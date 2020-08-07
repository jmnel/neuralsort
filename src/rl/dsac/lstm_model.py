from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 200
LSTM_LAYERS = 1


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):

    def __init__(self, state_space, device):
        super(Critic, self).__init__()

        self.state_space = state_space

        self.lstm = nn.LSTM(1, HIDDEN_SIZE, LSTM_LAYERS, batch_first=True, bias=False)

        self.linear1 = nn.Linear(HIDDEN_SIZE + 1, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, 2)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device))

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        x, a = state[:, :-1], state[:, -1:]
        x = x.unsqueeze(-1)
        x, self.hidden_cell = self.lstm(x, self.hidden_cell)

        x = x[:, -1, :]
#        x = F.tanh(x)
        z = torch.cat((x, a), dim=-1)
        z = self.linear1(self.elu1(z))
        z = self.linear2(self.elu2(z))
        z = self.linear3(z)

        return z

    def reset(self):
        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device))


class Actor(nn.Module):

    def __init__(self, state_space, device):
        super(Actor, self).__init__()

        self.state_space = state_space

        self.lstm = nn.LSTM(1, HIDDEN_SIZE, LSTM_LAYERS, batch_first=True, bias=False)

        self.linear1 = nn.Linear(HIDDEN_SIZE + 1, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, 2)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device))

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        x, a = state[:, :-1], state[:, -1:]
        x = x.unsqueeze(-1)

#        print(x)
#        print()

        x, self.hidden_cell = self.lstm(x, self.hidden_cell)

        x = x[:, -1, :]

#        x = F.tanh(x)
#        pprint(x)
#        print()
        z = torch.cat((x, a), dim=-1)
        z = self.linear1(self.elu1(z))
        z = self.linear2(self.elu2(z))
        z = self.linear3(z)

        return F.softmax(z, dim=-1)

    def reset(self):
        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device))
