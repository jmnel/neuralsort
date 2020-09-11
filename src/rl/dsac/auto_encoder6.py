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
#from tcn.wavenet_nn import WaveNetNN
from tcn import TemporalConvNet

BATCH_SIZE = 50
BATCH_SIZE_TRAIN = BATCH_SIZE
BATCH_SIZE_VALIDATE = BATCH_SIZE
BATCH_SIZE_TEST = BATCH_SIZE
# HIDDEN_SIZE = 50
# NUM_LAYERS = 2
SEQ_LEN = 2000
PRED_LEN = 100
BIAS = True
KERNEL = 2
DILATION = 3
NUM_LAYERS = 8
RESIDUAL_BLOCKS = 4
HIDDEN = 400
WEIGHT_DECAY = 1e-5
NUM_EXAMPLES = 1000
# PRE_HIDDEN = 1490
PRE_HIDDEN = 1408
LOSS_RECON_RETURN = 1
# LOSS_RECON_ABS = 1e-3
LOSS_RECON_ABS = 0
# ACTIVATION = torch.torch.tanh

EPOCHS = 5000

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True


def weights_init_(m):
    if isinstance(m, nn.Linear):
        #        torch.nn.init.sparse_(m.weight, sparsity=0.95, std=0.01)
        #        torch.nn.init.normal_(m.weight, 0, 1)
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
#        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
#        torch.nn.init.constant_(m.bias, 0)
#        torch.nn.init.normal_(m.weight, 0, 1)
#        torch.nn.init.normal_(m.bias, 0, 1)


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

#        self.tcn = WaveNetNN(layers=NUM_LAYERS,
#                             blocks=RESIDUAL_BLOCKS,
#                             dilation_channels=32,
#                             residual_channels=32,
#                             skip_channels=1,
#                             end_channels=1,
#                             classes=1,
#                             input_channels=1,
#                             output_channels=1,
#                             output_length=SEQ_LEN,
#                             kernel_size=2)

        self.tcn = TemporalConvNet(num_inputs=1, num_channels=[450] * (3 - 1) + [1], kernel_size=2, dropout=0.2)

        self.fc_1 = nn.Linear(2000, HIDDEN * 2)
        self.fc_2 = nn.Linear(HIDDEN * 2, HIDDEN)

#        self.skip_list = skip_list

#        self.input_layer = nn.Conv1d(1, 1, 1, bias=BIAS)

#        self.cnn_layers = nn.ModuleList()
#        self.norms = nn.ModuleList()

#        for layer_idx in range(NUM_LAYERS):
#            in_features = 2 ** layer_idx
#            out_features = 2 ** (layer_idx + 1)

#            dilation = DILATION**layer_idx
#            self.cnn_layers.append(nn.Conv1d(in_features, out_features, KERNEL, dilation=dilation, bias=BIAS))
#            self.norms.append(nn.BatchNorm1d(out_features))

#        self.hidden = nn.Linear(512 * 978, HIDDEN)
#        self.hidden = nn.Linear(2**NUM_LAYERS * PRE_HIDDEN, HIDDEN, bias=BIAS)

#        self.apply(weights_init_)
#        self.to(DEVICE)

    def forward(self, x):
        #        print(f'input: {x.shape}')
        x = self.tcn(x)
#        print(f'x: {x.shape}')
#        print(torch.max(x))
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
#        print(f'output: {x.shape}')
        return x


class RDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.tcn = TemporalConvNet(num_inputs=1, num_channels=[450] * (3 - 1) + [1], kernel_size=2, dropout=0.2)

        self.fc_1 = nn.Linear(HIDDEN, HIDDEN * 2)
        self.fc_2 = nn.Linear(HIDDEN * 2, 2000)

    def forward(self, x):

        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
        x = self.tcn(x)
#        print(f'x dec out: {x.shape}')

        return x

        # class PDecoder(nn.Module):

        #    def __init__(self, skip_list):
        #        super().__init__()

        #        self.skip_list = skip_list

        #        self.dcn_layers = nn.ModuleList()
        #        self.norms = nn.ModuleList()

        #        for layer_idx in range(NUM_LAYERS):
        #            in_features = 2 ** layer_idx
        #            out_features = 2 ** (layer_idx + 1)

        #            dilation = DILATION**layer_idx
        #            self.dcn_layers.append(nn.ConvTranspose1d(
        #                out_features, in_features, KERNEL, dilation=dilation, bias=BIAS))
        #            self.norms.append(nn.BatchNorm1d(in_features))

        #        self.output_layer = nn.Conv1d(1, 1, 1, bias=BIAS)

        #        self.output_fc = nn.Linear(SEQ_LEN, PRED_LEN, bias=BIAS)

        #        self.hidden = nn.Linear(HIDDEN, 2**NUM_LAYERS * PRE_HIDDEN, bias=BIAS)

        #        self.apply(weights_init_)
        #        self.to(DEVICE)

        #    def forward(self, x, skip):

        #        x = self.hidden(x)
        #        x = x.reshape(BATCH_SIZE, 2**NUM_LAYERS, PRE_HIDDEN)
        #        x = torch.tanh(x)

        #        for layer_idx in reversed(range(len(self.dcn_layers))):
        #            if self.skip_list[layer_idx]:
        #                x = x + skips_x[layer_idx]
        #            x = self.dcn_layers[layer_idx](x)

        #            if layer_idx != 0:
        #                x = self.norms[layer_idx](x)
        #                x = torch.tanh(x)

        #        x = self.output_layer(x)
        #        x = torch.tanh(x)
        #        x = self.output_fc(x)

        #        return x


class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

#        self.skip_list = list(idx >= SKIP_START for idx in range(NUM_LAYERS))

        self.encoder = Encoder()
        self.decoder_r = RDecoder()
#        self.decoder_p = PDecoder(self.skip_list)

#        self.apply(weights_init_)
        self.to(DEVICE)

    def forward(self, x):

        x = self.encoder(x)
        y_r = self.decoder_r(x)
#        y_p = self.decoder_p(x, skip)

        print(torch.max(y_r))

        return y_r


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

        y_r = model(x)

        re_cstr_loss = F.mse_loss(y_r, x[:, :, :], reduction='none')
        re_cstr_loss = torch.mean(re_cstr_loss * mask)

        z = torch.cat((torch.zeros((BATCH_SIZE_TRAIN, 1, 1)).to(DEVICE), y_r), axis=-1)
        z = torch.exp(torch.cumsum(z * std, dim=-1))

        loss_2 = F.mse_loss(z, q, reduction='mean')

        loss = LOSS_RECON_RETURN * re_cstr_loss + LOSS_RECON_ABS * loss_2
        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
    train_losses.append(avg_loss)

    print(f'train: {avg_loss}')


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

    with torch.no_grad():

        for batch_idx, (meta, x, mask) in enumerate(loader):

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

            y_r = model(x)

            re_cstr_loss = F.mse_loss(y_r, x[:, :, :], reduction='none')
            re_cstr_loss = torch.mean(re_cstr_loss * mask)

            z = torch.cat((torch.zeros((BATCH_SIZE_TRAIN, 1, 1)).to(DEVICE), y_r), axis=-1)
            z = torch.exp(torch.cumsum(z * std, dim=-1))

            loss_2 = F.mse_loss(z, q, reduction='mean')

            loss = LOSS_RECON_RETURN * re_cstr_loss + LOSS_RECON_ABS * loss_2

            validate_loss += loss

            if batch_idx == 0:
                reconstructed = (y_r[0, :].cpu().numpy())
                original = prices[0, :SEQ_LEN]
                original /= original[0]
                log_returns_ex = log_returns[0, :SEQ_LEN]

                log_returns_ex = log_returns_ex[:meta[2][0]]
                reconstructed = reconstructed[:meta[2][0]]
                real_len = meta[2][0]

                scale = std

        validate_loss *= BATCH_SIZE_TRAIN / len(loader.dataset)
        validate_losses.append(validate_loss)

    print(f'val: {validate_loss}')


loader = DataLoader(TickExamples(mode='train', num_examples=NUM_EXAMPLES),
                    batch_size=BATCH_SIZE_TRAIN,
                    shuffle=True,
                    drop_last=True)
model = Autoencoder().to(DEVICE)
optimizer = Adam(model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)

LAST_FILTER = 50


def main():
    for epoch in range(EPOCHS):

        train(model, loader, optimizer, epoch)

#        continue
        validate(model, loader)
#        plt.plot(original)
#        plt.show()

        fig, (ax1, ax4, ax2, ax3) = plt.subplots(4, 1)
        ax1.plot(np.arange(0, epoch + 1), train_losses)
        ax1.plot(np.arange(0, epoch + 1), validate_losses)
#        ax1.set_ylim((0, 4))

        ax4.plot(np.arange(0, min(epoch + 1, LAST_FILTER)), train_losses[-LAST_FILTER:])
        ax4.plot(np.arange(0, min(epoch + 1, LAST_FILTER)), validate_losses[-LAST_FILTER:])
#        ax1.set_ylim((0, 4))

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
