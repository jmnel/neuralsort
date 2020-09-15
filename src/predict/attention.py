from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import numpy as np

import settings
from rl.dsac.tick_examples import TickExamples

EPOCHS = 1
HIDDEN_SIZE = 2
M = 5

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1, HIDDEN_SIZE, 1, batch_first=True)

        self.cell_state = (torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
                           torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE))

        self.weight_2 = nn.Parameter(torch.zeros(10, M))
        self.weight_1 = nn.Parameter(torch.zeros(10, HIDDEN_SIZE * 2))
        self.v = nn.Parameter(torch.zeros(1, 10))

    def forward(self, x_in):

        x, self.cell_state = self.lstm(x_in, self.cell_state)

        print(f'x: {x.shape}')
        print(f'h: {self.cell_state[0].shape} {self.cell_state[1].shape}')
        exit()

        print(self.cell_state[1].shape)

        z = torch.cat(self.cell_state, axis=-1).reshape((1, 2 * HIDDEN_SIZE, 1))
        print(f'in: {x_in.shape}')

        x_2 = torch.matmul(self.weight_2, x_in)
#        x_1 = torch.matmul(self.weight_1, torch.cat(self.cell_state, axis=1))

        x_1 = torch.matmul(self.weight_1, z)

        alpha = torch.matmul(self.v, torch.tanh(x_1 + x_2))
        print(alpha.shape)
#        y = self.weight_2 * x
#        print(y.shape)

        return x
#        self.attention = nn.Parameter(torch.empty(


train_losses = list()


def train(model, loader, optmizer, epoch):

    loader.dataset.set_mode('train')
    model.train()
    avg_loss = 0

    for batch_idx, (labels, ticks, mask) in enumerate(loader):

        optimizer.zero_grad()

        seq_len = labels[2][0]
        ticks = ticks[:, :seq_len, :]
#        print(ticks.shape)
        prices = ticks[:, :, 1]
        x = np.diff(np.log(prices))
        x = torch.from_numpy(x)
        x = x.reshape((*x.shape, 1))
        x = x.to(DEVICE)
#        print(x.shape)

#        r = tuple(range(0, x.shape[1] - M))
        for t in range(0, x.shape[1] - M):
            y = model(x[:, t: t + M, :])
            model.cell_state = (torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
                                torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE))
            exit()
#            print(f'{t} : 0 -> {x.shape[1]}')
            #            assert False
            #            print(t)

            #            model.cell_state = (torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
            #                                torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE))

            #            y = model(x)


loader = DataLoader(dataset=TickExamples(num_examples=1),
                    batch_size=1,
                    shuffle=False,
                    drop_last=True)
model = Model()
model = model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    train(model, loader, optimizer, epoch)
