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

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)

        self.batch1 = nn.BatchNorm1d(4)
        self.batch2 = nn.BatchNorm1d(16)
        self.batch3 = nn.BatchNorm1d(32)

        self.linear1 = nn.Linear(320 + 1, 128)
        self.linear2 = nn.Linear(128, 2)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        x, a = state[:, :-1], state[:, -1:]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.elu1(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.elu2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.elu3(x)

        x = x.flatten(start_dim=1)
        z = torch.cat((x, a), dim=-1)
        z = self.elu4(self.linear1(z))
        z = self.linear2(z)

        return z

    def reset(self):
        pass


class Actor(nn.Module):

    def __init__(self, state_space, device):
        super(Actor, self).__init__()

        self.state_space = state_space

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)

        self.batch1 = nn.BatchNorm1d(4)
        self.batch2 = nn.BatchNorm1d(16)
        self.batch3 = nn.BatchNorm1d(32)

        self.linear1 = nn.Linear(320 + 1, 128)
        self.linear2 = nn.Linear(128, 2)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        x, a = state[:, :-1], state[:, -1:]
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.elu1(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.elu2(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.elu3(x)

        x = x.flatten(start_dim=1)
        z = torch.cat((x, a), dim=-1)
        z = self.elu4(self.linear1(z))
        z = self.linear2(z)

        return F.softmax(z, dim=-1)

    def reset(self):
        pass
