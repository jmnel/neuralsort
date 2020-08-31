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
        self.device = device

        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)

        self.batch1 = nn.BatchNorm1d(16)
        self.batch2 = nn.BatchNorm1d(32)
        self.batch3 = nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(3072, 512)
        self.linear2 = nn.Linear(512, 3)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        #        print(self.device)
        #        x, a = state[:, :-1], state[:, -1:]
        #        x = x.unsqueeze(1)

        #        print(state.shape)
        #        print(x.shape)
        x = state.reshape((state.shape[0], state.shape[2], state.shape[1]))

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
        x = self.elu4(self.linear1(x))
        x = self.linear2(x)

        return x

    def reset(self):
        pass


class Actor(nn.Module):

    def __init__(self, state_space, device):
        super(Actor, self).__init__()

        self.state_space = state_space
        self.device = device

        self.conv1 = nn.Conv1d(in_channels=10, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2)

        self.batch1 = nn.BatchNorm1d(16)
        self.batch2 = nn.BatchNorm1d(32)
        self.batch3 = nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(3072, 512)
        self.linear2 = nn.Linear(512, 3)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

        self.apply(weights_init_)

        self.to(device)

    def forward(self, state):

        #        print(state.shape)
        #        x, a = state[:, :-1], state[:, -1:]
        #        x = x.unsqueeze(1)

        x = state.reshape((state.shape[0], state.shape[2], state.shape[1]))

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

#        print(x.shape)

        x = self.elu4(self.linear1(x))
        x = self.linear2(x)

        return F.softmax(x, dim=-1)

    def reset(self):
        pass
