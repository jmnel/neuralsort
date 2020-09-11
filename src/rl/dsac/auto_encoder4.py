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

BATCH_SIZE = 50
BATCH_SIZE_TRAIN = BATCH_SIZE
BATCH_SIZE_VALIDATE = BATCH_SIZE
BATCH_SIZE_TEST = BATCH_SIZE
# HIDDEN_SIZE = 50
# NUM_LAYERS = 2
SEQ_LEN = 2000
BIAS = False
KERNEL = 3
DILATION = 2
NUM_LAYERS = 9
SKIP_START = 90
HIDDEN = 100
WEIGHT_DECAY = 0.0
NUM_EXAMPLES = 10000
#ACTIVATION = torch.torch.tanh

EPOCHS = 5000

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

torch.backends.cudnn.benchmark = True


def weights_init_(m):
    if isinstance(m, nn.Linear):
        #        torch.nn.init.sparse_(m.weight, sparsity=0.95, std=0.01)
        #        torch.nn.init.normal_(m.weight, 0, 1)
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
#        torch.nn.init.normal_(m.weight, 0, 1)
#        torch.nn.init.normal_(m.bias, 0, 1)


class ModelA(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn_in = nn.Conv1d(1, 1, 1, bias=BIAS)
#        self.elu_in = nn.ELU()

        self.cnn_layers = list()
        self.dcn_layers = list()
        self.norms_1 = list()
        self.norms_2 = list()

#        self.elus_1 = nn.ModuleList(nn.ELU() for _ in range(NUM_LAYERS))
#        self.elus_2 = nn.ModuleList(nn.ELU() for _ in range(NUM_LAYERS))

        for layer_idx in range(NUM_LAYERS):
            in_features = 2 ** layer_idx
            out_features = 2 ** (layer_idx + 1)

            dilation = DILATION**layer_idx
            self.cnn_layers.append(nn.Conv1d(in_features, out_features, KERNEL, dilation=dilation, bias=BIAS))
            self.norms_1.append(nn.BatchNorm1d(out_features))
            self.dcn_layers.append(nn.ConvTranspose1d(
                out_features, in_features, KERNEL, dilation=dilation, bias=BIAS))
            self.norms_2.append(nn.BatchNorm1d(in_features))

        self.cnn_layers = nn.ModuleList(self.cnn_layers)
        self.dcn_layers = nn.ModuleList(self.dcn_layers)
        self.norms_1 = nn.ModuleList(self.norms_1)
        self.norms_2 = nn.ModuleList(self.norms_2)

        self.cnn_out = nn.Conv1d(1, 1, 1, bias=BIAS)

#        self.elu_h_1 = nn.ELU()
#        self.elu_h_2 = nn.ELU()

        self.hidden_1 = nn.Linear(978, HIDDEN)
        self.hidden_2 = nn.Linear(HIDDEN, 978)

        self.apply(weights_init_)

#        bias = torch.empty((BATCH_SIZE_TRAIN, 1, SEQ_LEN),
#                           device=DEVICE, requires_grad=True)
#        self.bias = nn.Parameter(bias)
#        torch.nn.init.normal_(self.bias)

        self.to(DEVICE)

    hidden = 0

    def forward(self, x):

        global hidden

        x = self.cnn_in(x)
        x = torch.tanh(x)
#        x = self.elu_in(x)

        skips = list(idx >= SKIP_START for idx in range(NUM_LAYERS))
#        skips[3] = True

#        skips[2] = True
        skips_x = list(None for _ in range(NUM_LAYERS))

        for layer_idx in range(len(self.cnn_layers)):
            x = self.cnn_layers[layer_idx](x)

            if skips[layer_idx]:
                skips_x[layer_idx] = x.clone()
#            if layer_idx + 1 != len(self.cnn_layers):

            if layer_idx + 1 != NUM_LAYERS:
                x = self.norms_1[layer_idx](x)
#                x = self.elus_1[layer_idx](x)
                x = torch.tanh(x)
#            x = F.relu(x)

        x = self.hidden_1(x)
        x = torch.tanh(x)
#        x = self.elu_h_1(x)
        x = self.hidden_2(x)
        x = torch.tanh(x)
#        x = self.elu_h_2(x)
        hidden = x.shape[-1]
#        print(hidden)

        for layer_idx in reversed(range(len(self.dcn_layers))):
            if skips[layer_idx]:
                x = x + skips_x[layer_idx]
            x = self.dcn_layers[layer_idx](x)

            if layer_idx != 0:
                x = self.norms_2[layer_idx](x)
                x = torch.tanh(x)
#                x = self.elus_2[layer_idx](x)

#            x = F.
#            x = F.relu(x)

#        x = self.cnn_out(x)

#        x = x + self.bias

#        print(torch.max(x))

#        print(self.cnn_layers[0]._parameters['weight'])
#        print(self.cnn_layers[0]._parameters['bias'])

        return x


train_losses = list()


def train(model, loader: DataLoader, optimizer, epoch):

    loader.dataset.set_mode('train')

    model.train()
    avg_loss = 0

    for batch_idx, (_, x, mask) in enumerate(loader):

        optimizer.zero_grad()

        x = x.numpy()
        prices = x[:, :SEQ_LEN + 1, 1]
        mask = mask[:, :SEQ_LEN + 1]

        prices += (1 - mask.numpy())

        mask = mask[:, 1:]

        log_returns = np.diff(np.log(prices), axis=-1)

#        mu = np.mean(log_returns, axis=1).reshape((BATCH_SIZE_TRAIN, 1))
#        mu = np.repeat(mu, SEQ_LEN, -1)

#        log_returns = log_returns - mu
        std = np.std(log_returns, axis=-1).reshape((BATCH_SIZE_TRAIN, 1))
        std = np.mean(std)
#        std = np.repeat(std, SEQ_LEN, -1)
        log_returns = log_returns / std

        q = torch.from_numpy(prices)
        q = q.to(DEVICE)
        q = q.reshape((BATCH_SIZE_TRAIN, 1, q.shape[1]))

        x = log_returns

        x = x.reshape((BATCH_SIZE_TRAIN, 1, x.shape[1]))
        mask = mask.to(DEVICE)
        mask = mask.reshape((BATCH_SIZE_TRAIN, 1, mask.shape[-1]))
        x = torch.from_numpy(x)
        x = x.to(DEVICE)
        x = x * mask

        y = model1(x)

        re_cstr_loss = F.mse_loss(y, x[:, :, :], reduction='none')
        re_cstr_loss = torch.mean(re_cstr_loss * mask)

#        print(f's1: {y.shape}')
#        z = torch.cat((torch.zeros((BATCH_SIZE_TRAIN, 1, 1)).to(DEVICE), y * 1e-3), axis=-1)
#        z = torch.exp(torch.cumsum(z, dim=-1))

#        loss_2 = F.mse_loss(z, q, reduction='mean')

#        print(re_cstr_loss)
#        print(loss_2)
#        print('-' * 10)

        loss = re_cstr_loss  # + loss_2 * 0e-4
        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
    train_losses.append(avg_loss)

#    print(avg_loss)


validate_losses = list()

reconstructed = None
actual = None
original = None

rcstr_log_returns_ex = None
log_returns_ex = None

real_len = 0
scale = 1.0


def validate(model, loader):

    loader.dataset.set_mode('validate')
    model.eval()
    validate_loss = 0.

    global actual
    global reconstructed
    global original

    global log_returns_ex
    global rcstr_log_returns_ex
    global real_len
    global scale

    print(f'hidden: {hidden}')

    with torch.no_grad():

        for batch_idx, (meta, x, mask) in enumerate(loader):

            #            print(meta)
            #            exit()
            x = x.numpy()
            prices = x[:, :SEQ_LEN + 1, 1]
            mask = mask[:, :SEQ_LEN + 1]

            prices += (1 - mask.numpy())

            mask = mask[:, 1:]

            log_returns = np.diff(np.log(prices), axis=-1)

            mu = np.mean(log_returns, axis=1).reshape((BATCH_SIZE_TRAIN, 1))
            mu = np.repeat(mu, SEQ_LEN, -1)

#            log_returns = log_returns - mu
            std = np.std(log_returns, axis=-1).reshape((BATCH_SIZE_TRAIN, 1))
            std = np.mean(std)
#            std = np.repeat(std, SEQ_LEN, -1)
            log_returns = log_returns / std

            q = torch.from_numpy(prices)
            q = q.to(DEVICE)
            q = q.reshape((BATCH_SIZE_TRAIN, 1, q.shape[1]))

            x = log_returns

            x = x.reshape((BATCH_SIZE_TRAIN, 1, x.shape[1]))
            mask = mask.to(DEVICE)
            mask = mask.reshape((BATCH_SIZE_TRAIN, 1, mask.shape[-1]))
            x = torch.from_numpy(x)
            x = x.to(DEVICE)
            x = x * mask

            y = model1(x)

            re_cstr_loss = F.mse_loss(y, x[:, :, :], reduction='none')
            re_cstr_loss = torch.mean(re_cstr_loss * mask)
#            z = torch.cat((torch.zeros((BATCH_SIZE_TRAIN, 1, 1)).to(DEVICE), y), axis=-1)
#            z = torch.exp(torch.cumsum(z, dim=-1))

#            loss_2 = F.mse_loss(z, q, reduction='mean')

            loss = re_cstr_loss  # + 0e-4 * loss_2

            validate_loss += loss

            if batch_idx == 0:
                reconstructed = (y[0, :].cpu().numpy())
                original = prices[0, :SEQ_LEN]
                original /= original[0]
                log_returns_ex = log_returns[0, :SEQ_LEN]

                log_returns_ex = log_returns_ex[:meta[2][0]]
                reconstructed = reconstructed[:meta[2][0]]
                real_len = meta[2][0]

                scale = std

        validate_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
        validate_losses.append(validate_loss)

#        print(f'valid: {validate_loss}')


loader = DataLoader(TickExamples(mode='train', num_examples=NUM_EXAMPLES),
                    batch_size=BATCH_SIZE_TRAIN,
                    shuffle=True,
                    drop_last=True)
model1 = ModelA().to(DEVICE)
optimizer = Adam(model1.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)

LAST_FILTER = 50


def main():
    for epoch in range(EPOCHS):

        train(model1, loader, optimizer, epoch)

#        continue
        validate(model1, loader)
#        plt.plot(original)
#        plt.show()

        fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1)
        ax1.plot(np.arange(0, epoch + 1), train_losses)
        ax1.plot(np.arange(0, epoch + 1), validate_losses)

        ax4.plot(np.arange(0, min(epoch + 1, LAST_FILTER)), train_losses[-LAST_FILTER:])
        ax4.plot(np.arange(0, min(epoch + 1, LAST_FILTER)), validate_losses[-LAST_FILTER:])

        ax2.plot(np.arange(log_returns_ex.shape[0] - 1), log_returns_ex[:-1])
        ax2.plot(np.arange(reconstructed[0, :real_len - 1].shape[0]), reconstructed[0, :real_len - 1])
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
        p = np.concatenate(((1.,), np.exp(np.cumsum(reconstructed * scale))))

#        print(f'o: {original.shape}')

        ax3.plot(np.arange(original.shape[0])[:real_len - 1], original[:real_len - 1])
        ax3.plot(np.arange(p.shape[0])[:real_len - 1], p[:real_len - 1])
#            ax3.plot(np.arange(len(actual_t)), actual_t)
#            ax3.plot(np.arange(recon_t.shape[0]), recon_t)

        plt.show()


if __name__ == '__main__':
    main()
