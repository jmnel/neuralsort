import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 200
LSTM_LAYERS = 1
LSTM_FEATUES = 10


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Critic(nn.Module):

    def __init__(self, state_space, device):
        super(Critic, self).__init__()

        self.state_space = state_space
        self.device = device

        self.lstm = nn.LSTM(LSTM_FEATUES, HIDDEN_SIZE, LSTM_LAYERS, batch_first=True, bias=False)

        self.linear1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, 3)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(device))

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        #        state_env, state_agent = state

        x, self.hidden_cell = self.lstm(state, self.hidden_cell)

        x = x[:, -1, :]
#        state_agent = state_agent.unsqueeze(dim=0)
#        z = torch.cat((state_env, state_agent), dim=-1)
        z = self.linear1(self.elu1(x))
        z = self.linear2(self.elu2(z))
        z = self.linear3(z)

        return z

    def reset(self):
        print('reset1')
#        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device),
#                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device))


class Actor(nn.Module):

    def __init__(self, state_space, device):
        super(Actor, self).__init__()

        self.state_space = state_space
        self.device = device

        self.lstm = nn.LSTM(LSTM_FEATUES, HIDDEN_SIZE, LSTM_LAYERS, batch_first=True, bias=False)

        self.linear1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = nn.Linear(HIDDEN_SIZE, 3)

        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device),
                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device))

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        #        state_env, state_agent = state

        #        print(f'state: {state.shape}')

        x, self.hidden_cell = self.lstm(state, self.hidden_cell)

        x = x[:, -1, :]

#        state_agent = state_agent.unsqueeze(dim=0)
#        z = torch.cat((x, state_agent), dim=-1)

        z = self.linear1(self.elu1(x))
        z = self.linear2(self.elu2(z))
        z = self.linear3(z)

        return F.softmax(z, dim=-1)

    def reset(self):
        print('reset2')
#        self.hidden_cell = (torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device),
#                            torch.zeros(LSTM_LAYERS, 1, HIDDEN_SIZE).to(self.device))
