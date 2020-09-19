from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import settings
from rl.dsac.tick_examples import TickExamples
import gramian_field as gf

DEVICE = torch.device('cuda')
NUM_LAYERS = 4
EPOCHS = 10000
#BATCH_SIZE = 40
BATCH_SIZE = 10
NUM_EXAMPLES = 100
HIDDEN = 64
PADDING = 0
LR = 2e-3
LOSS_2 = 2
BIAS = False
ACT = torch.tanh


def weights_init_(m):
    if isinstance(m, nn.Linear):
        #        torch.nn.init.sparse_(m.weight, sparsity=0.95, std=0.01)
        #        torch.nn.init.normal_(m.weight, 0, 1)
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
#        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
#        torch.nn.init.constant_(m.bias, 0)
#        torch.nn.init.normal_(m.weight, 0, 1)
#        torch.nn.init.normal_(m.bias, 0, 1)


class Model(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.gram = gf.GASF(device)

        self.conv_in = nn.Conv2d(1, 2**NUM_LAYERS, 8, bias=BIAS)

        self.norms_1 = nn.ModuleList()

#        self.elu_1 = nn.ModuleList()

        self.conv_layers = nn.ModuleList()
        final_size = 512 + 2 * PADDING - 8 + 1
        for layer_idx in range(NUM_LAYERS):
            i = NUM_LAYERS - layer_idx
            self.conv_layers.append(nn.Conv2d(2**i, 2**(i - 1), 8, 1, bias=BIAS))
            self.norms_1.append(nn.BatchNorm2d(2**(i - 1)))
#            self.elu_1.append(nn.ELU())
            final_size = (final_size - 8 + 1)
            if layer_idx + 1 < NUM_LAYERS:
                final_size = final_size // 2

        out_size = final_size ** 2

        self.fc_1 = nn.Linear(out_size, HIDDEN * 2, bias=BIAS)
        self.fc_2 = nn.Linear(HIDDEN * 2, HIDDEN, bias=BIAS)

        self.dconv_layers = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
#        self.elu_2 = nn.ModuleList()
        for layer_idx in range(NUM_LAYERS):
            i = layer_idx
            self.dconv_layers.append(nn.ConvTranspose2d(2**i, 2**(i + 1), 8, 1, bias=BIAS))
            self.norms_2.append(nn.BatchNorm2d(2**(i + 1)))
#            self.elu_2.append(nn.ELU())

        self.fc_3 = nn.Linear(HIDDEN, HIDDEN * 2, bias=BIAS)
        self.fc_4 = nn.Linear(HIDDEN * 2, out_size, bias=BIAS)

        self.dconv_out = nn.ConvTranspose2d(2**NUM_LAYERS, 1, 8, bias=BIAS)

        self.conv_out_1 = nn.Conv2d(1, 1, 1, bias=BIAS)
        self.conv_out_2 = nn.Conv2d(1, 1, 1, bias=BIAS)

        self.gram_inverse = gf.InverseGASF(device)

#        self.conv_f1 = nn.Conv2d(1, 4, kernel_size=2, padding=1)
#        self.conv_f2 = nn.Conv2d(4, 1, kernel_size=2, padding=0)

        self.apply(weights_init_)

        self.to(device)

    def forward(self, x_in):
        gram, scale_min, scale_max = self.gram(x_in)

        x = torch.clone(gram)

#        print(torch.mean(x))

#        print(torch.max(x))
#        print(torch.min(x))
#        p0 = torch.zeros(BATCH_SIZE, 1, PADDING, 508).to(self.device)
#        p1 = torch.zeros(BATCH_SIZE, 1, 512, PADDING).to(self.device)

#        print(x.shape)
#        x = torch.cat((p0, x, p0), dim=-2)
#        x = torch.cat((p1, x, p1), dim=-1)
#        print(f'paddded: {x.shape}')

        x = self.conv_in(x)
        x = ACT(x)

        pool_indices = list()

        for layer_idx in range(NUM_LAYERS):
            x = self.conv_layers[layer_idx](x)

            if layer_idx + 2 < NUM_LAYERS:
                x = self.norms_1[layer_idx](x)
#            x = self.elu_1[layer_idx](x)
            x = ACT(x)
            if layer_idx + 1 < NUM_LAYERS:
                x, indices = F.max_pool2d(x, kernel_size=2, return_indices=True, stride=2)
                pool_indices.append(indices)

#        print(f'{x.shape}')
        x = torch.flatten(x, start_dim=1)

        x = self.fc_1(x)
        x = ACT(x)
        x = self.fc_2(x)

        x = self.fc_3(x)
        x = ACT(x)
        x = self.fc_4(x)

        x = x.reshape(x.shape[0], 1, 50, 50)

        for layer_idx in range(NUM_LAYERS):
            if layer_idx > 0:
                x = F.max_unpool2d(x, indices=pool_indices[-layer_idx], kernel_size=2, stride=2)
            x = self.dconv_layers[layer_idx](x)

            if layer_idx + 2 < NUM_LAYERS:
                x = self.norms_2[layer_idx](x)
            x = ACT(x)
#            x = self.elu_2[layer_idx](x)

        x = self.dconv_out(x)
#        x = ACT(x)

#        x_res = x
#        x = self.conv_out_1(x)
#        x = ACT(x)
#        x = self.conv_out_2(x) + x_res


#        x_res = x
#        x = self.conv_f1(x)
#        x = ACT(x)
#        x = self.conv_f2(x) + x_res

        x_rcst = self.gram_inverse(x, scale_min, scale_max)

#        x = x[:, :, 2:510, 2:510]
#        print(f'out: {x.shape}')

        return x, gram, scale_min, scale_max, x_rcst


train_losses = list()


def train(model, loader: DataLoader, optimizer, epoch):

    loader.dataset.set_mode('train')

    model.train()
    avg_loss = 0

    for batch_idx, (labels, ticks, mask) in enumerate(loader):

        optimizer.zero_grad()
        ticks = ticks.numpy()
        price = ticks[:, :513 - 2 * PADDING, 1:2]
#        print(price.shape)
#        price = price / price[:, 0, :]
        for i in range(BATCH_SIZE):
            price[i, :, :] = price[i, :, :] / price[i, 0, 0]
        price = price.reshape(BATCH_SIZE, 1, 513 - 2 * PADDING)

        log_returns = np.diff(np.log(price), axis=-1)
        x = torch.from_numpy(log_returns).to(DEVICE)
#        x = torch.from_numpy(np.log(price[:, :, :512])).to(DEVICE)

        y, gram, scale_min, scale_max, x_rcst = model(x)

#        loss_2 = F.mse_loss(x_rcst, x[:, 0, :])
        loss = F.mse_loss(y, gram)

        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= BATCH_SIZE / len(loader.dataset)
    train_losses.append(avg_loss)

    print(f'Train loss: {avg_loss}')


validate_losses = list()

plt_gram = None
plt_gram_hat = None
plt_x_rcst = None
plt_x = None


def validate(model, loader):

    global plt_gram, plt_gram_hat, plt_x_rcst, plt_x

    loader.dataset.set_mode('validate')
    model.eval()
    validate_loss = 0

    with torch.no_grad():
        for batch_idx, (labels, ticks, mask) in enumerate(loader):
            ticks = ticks.numpy()
            price = ticks[:, :513 - 2 * PADDING, 1:2]
            price = price.reshape(BATCH_SIZE, 1, 513 - 2 * PADDING)

#            log_returns = np.diff(np.log(price), axis=-1)
#            x = torch.from_numpy(log_returns).to(DEVICE)
            for i in range(BATCH_SIZE):
                price[i, :, :] = price[i, :, :] / price[i, 0, 0]

            log_returns = np.diff(np.log(price), axis=-1)
            x = torch.from_numpy(log_returns).to(DEVICE)
#            x = torch.from_numpy(np.log(price[:, :, :512])).to(DEVICE)
            y, gram, scale_min, scale_max, x_rcst = model(x)

#            loss_2 = F.mse_loss(x_rcst, x[:, 0, :])
            loss = F.mse_loss(y, gram)
            validate_loss += loss

            if batch_idx == 0:
                plt_gram_hat = gram[0, 0].cpu().numpy().astype(float)
                plt_gram = y[0, 0].cpu().numpy().astype(float)
                plt_x_rcst = x_rcst[0, :].cpu().numpy().astype(float)
                plt_x = x[0, 0, :].cpu().numpy().astype(float)
#                print(plt_x.shape)

    validate_loss *= BATCH_SIZE / len(loader.dataset)
    validate_losses.append(validate_loss)
    print(f'Validate loss: {validate_loss}')


loader = DataLoader(dataset=TickExamples(num_examples=NUM_EXAMPLES),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True)
model = Model(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)


def main():

    for epoch in range(EPOCHS):

        train(model, loader, optimizer, epoch)
        validate(model, loader)

        fig, axes = plt.subplots(3, 2)
        if plt_gram is not None and plt_gram_hat is not None:
            axes[0][0].imshow(plt_gram_hat)
            axes[0][1].imshow(plt_gram[:, :])
        axes[1][0].plot(train_losses)
        axes[1][0].plot(validate_losses)
#        print(plt_x.shape)
#        print(plt_x_rcst.shape)
        axes[1][1].plot(plt_x)
        axes[1][1].plot(plt_x_rcst)
#        print(plt_x_rcst)
#        axes[2][0].plot(np.exp(np.cumsum(plt_x)))
#        axes[2][0].plot(np.exp(np.cumsum(plt_x_rcst)))

        axes[2][1].plot(plt_x_rcst)
        plt.show()


if __name__ == '__main__':
    main()
