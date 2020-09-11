from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import numpy as np
import random

import settings
from tick_examples import TickExamples

BATCH_SIZE_TRAIN = 1
BATCH_SIZE_VALIDATE = 1
BATCH_SIZE_TEST = 1
HIDDEN_SIZE = 50
NUM_LAYERS = 2
SEQ_LEN = 400
LSTM_BIAS = True

EPOCHS = 500

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class ModelA(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm_1 = nn.LSTM(input_size=1,
                              hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS,
                              batch_first=True,
                              bias=LSTM_BIAS)
        self.lstm_2 = nn.LSTM(input_size=HIDDEN_SIZE,
                              hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS,
                              batch_first=True,
                              bias=LSTM_BIAS)

        self.hidden_cell_1 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                              torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))
        self.hidden_cell_2 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                              torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))

        self.fc_1 = nn.Linear(in_features=HIDDEN_SIZE * SEQ_LEN, out_features=HIDDEN_SIZE * SEQ_LEN // pow(2, 2))
        self.fc_2 = nn.Linear(HIDDEN_SIZE * SEQ_LEN // pow(2, 2), SEQ_LEN)

        self.to(DEVICE)

    def forward(self, x):

        x, self.hidden_cell_1 = self.lstm_1(x, self.hidden_cell_1)

        x = x[:, -1:, :].repeat(1, SEQ_LEN, 1)

        x, self.hidden_cell_2 = self.lstm_2(x, self.hidden_cell_2)

        x = x.flatten(start_dim=1)

        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)

        return x


train_losses = list()


def train(model, loader: DataLoader, optimizer, epoch):

    loader.dataset.set_mode('train')

    model.train()
    avg_loss = 0

    for batch_idx, (_, x, mask) in enumerate(loader):

        optimizer.zero_grad()

        x = x.numpy()
        ts = x[:, :-1, 0]
        prices = x[:, :, 1]

        for i in range(prices.shape[1]):
            if prices[0, i] == 0.0:
                prices[0, i] = 1e-9

        sizes = x[:, :-1, 2]
        log_returns = np.diff(np.log(prices), axis=-1) * 5e2
        x = np.vstack((ts, log_returns, sizes))

        x = x.reshape((BATCH_SIZE_TRAIN, x.shape[1], 3))
#        x = x.reshape((1, x.shape[0], 3))
#        [:, :-1, 1] = np.diff(np.log(x[:, :, 1]), axis=1)
        x = torch.from_numpy(x)

        x = x[:, :SEQ_LEN, 1:2]
        x = x.to(DEVICE)

        model1.hidden_cell_1 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                                torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))
        model1.hidden_cell_2 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                                torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))

        y = model1(x)

        re_cstr_loss = F.mse_loss(y, x[:, :, 0])
        loss = re_cstr_loss
        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
    train_losses.append(avg_loss)

    print(avg_loss)


validate_losses = list()

reconstructed = None
actual = None
original = None

rcstr_log_returns_ex = None
log_returns_ex = None


def validate(model, loader):
    loader.dataset.set_mode('validate')
    model.eval()
    validate_loss = 0.

    global actual
    global reconstructed
    global original

    global log_returns_ex
    global rcstr_log_returns_ex

    with torch.no_grad():

        for batch_idx, (_, x, mask) in enumerate(loader):

            x = x.numpy()
            ts = x[:, :-1, 0]
            prices = x[:, :, 1]

            for i in range(prices.shape[1]):
                if prices[0, i] == 0.0:
                    prices[0, i] = 1e-9

            sizes = x[:, :-1, 2]

            log_returns = np.diff(np.log(prices), axis=-1) * 5e2

            x = np.vstack((ts, log_returns, sizes))

            x = x.reshape((BATCH_SIZE_TRAIN, x.shape[1], 3))
            x = torch.from_numpy(x)

            x = x[:, :SEQ_LEN, 1:2]
            x = x.to(DEVICE)

            model1.hidden_cell_1 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                                    torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))
            model1.hidden_cell_2 = (torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE),
                                    torch.zeros(NUM_LAYERS, BATCH_SIZE_TRAIN, HIDDEN_SIZE).to(DEVICE))

            y = model1(x)


#            if batch_idx == random.randint(0, len(loader.dataset) - 1):
            if batch_idx == 3:

                #                actual = x[0, :, 1].cpu().numpy()
                reconstructed = y[0, :].cpu().numpy()
                original = prices[0, :SEQ_LEN]
                original /= original[0]
                log_returns_ex = log_returns[0, :SEQ_LEN]
#                rcstr_log_returns_ex = y[0, :] * 5e-2
#                plt.plot(original)
#                plt.plot(original)
#                plt.show()
#                exit()
#                plt.show()

            re_cstr_loss = F.mse_loss(y, x[:, :, 0])
            loss = re_cstr_loss

            validate_loss += loss

        validate_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
        validate_losses.append(validate_loss)

        print(f'valid: {validate_loss}')


loader = DataLoader(TickExamples(mode='train'),
                    batch_size=BATCH_SIZE_TRAIN,
                    shuffle=False)
model1 = ModelA().to(DEVICE)
optimizer = Adam(model1.parameters(), lr=1e-3)


def main():
    for epoch in range(EPOCHS):

        train(model1, loader, optimizer, epoch)
        validate(model1, loader)
#        plt.plot(original)
#        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(np.arange(0, epoch + 1), train_losses)
        ax1.plot(np.arange(0, epoch + 1), validate_losses)

        ax2.plot(np.arange(log_returns_ex.shape[0]), log_returns_ex)
        ax2.plot(np.arange(log_returns_ex.shape[0]), reconstructed)
#        ax2.plot(np.arange(original.shape[0]), original)
#        if actual is not None:
        #            ax2.plot(np.arange(actual.shape[0]), actual)
        #            ax2.plot(np.arange(reconstructed.shape[0]), reconstructed)

        #            actual_t = [1.0, ]
        #            for i in range(len(actual)):
        #                actual_t.append((actual_t[-1]) * np.exp(actual[i]))
        #                print(actual[i])
        #            actual_t = np.exp(actual.cumsum())
        #            actual_t = np.concatenate(((1,), actual_t))
        #            recon_t = np.exp(reconstructed.cumsum())
        #            recon
        #            recon_t = np.concatenate(((1,), recon_t))

#        print(reconstructed.shape)
#        o = np.diff(np.log())
        p = np.concatenate(((1.,), np.exp(np.cumsum(reconstructed * 5e-2))))

        print(f'o: {original.shape}')

        ax3.plot(np.arange(original.shape[0]), original)
        ax3.plot(np.arange(p.shape[0]), p)
#            ax3.plot(np.arange(len(actual_t)), actual_t)
#            ax3.plot(np.arange(recon_t.shape[0]), recon_t)

        plt.show()


if __name__ == '__main__':
    main()
