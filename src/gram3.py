from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import settings
from rl.dsac.tick_examples import TickExamples
import gramian_field as gf
import paa
from squeeze_excite import SeBlock2d

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

DEVICE = torch.device('cuda')
#NUM_LAYERS = 6
EPOCHS = 10000
BATCH_SIZE = 50
#BATCH_SIZE = 10
NUM_EXAMPLES = 1000
HIDDEN = 8
PADDING = 0
LR = 1e-2
LOSS_2 = 1
LOSS_1 = 1
LOSS_3 = 1
BIAS = True
#ACT = torch.tanh
ACT = F.relu
RESIDUAL_BLOCKS = 6
RES_DEPTH = 3
CHANNELS = 16
BLOCK_MIN = 80
BLOCK_MAX = 351
DILATION = 1
ALPHA = 0.5
L = 5
SEQ_LEN = 255
DROPOUT = 0.0
SE_RATIO = 2


def calc_ema(x, alpha, l):
    ema = np.zeros_like(x)
    ema[:, 0, 0] = x[:, 0, 0]
    for t in range(1, x.shape[1]):
        if t != 0:
            ema[:, t, 0] += x[:, t, 0] * (ALPHA / (1 + L))
            ema[:, t, 0] += ema[:, t - 1, 0] * (1 - ALPHA / (1 + L))
    return ema


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


class ResidualBlock(nn.Module):

    def __init__(self, in_features, num_layers):
        super().__init__()

        self.in_features = in_features
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()

        self.norm = nn.BatchNorm2d(in_features)

        for i in range(num_layers):
            f_in = in_features
            f_out = f_in
            self.conv_layers.append(nn.Conv2d(f_in, f_out, kernel_size=7, padding=3 *
                                              DILATION, bias=BIAS, dilation=DILATION))

        self.pool_indices = None

    def forward(self, x):

        x_res = x
        for i in range(self.num_layers):
            s1 = x.shape
            x = self.conv_layers[i](x)
            x = ACT(x)
#            print(f'{i} : {s1} -> {x.shape}')
        x = x + x_res
        x = self.norm(x)
        x, pool_indices = F.max_pool2d(x, kernel_size=2, return_indices=True, stride=2)
        return x, pool_indices


class DeResidualBlock(nn.Module):

    def __init__(self, in_features, num_layers):
        super().__init__()

        self.in_features = in_features
        self.num_layers = num_layers

        self.dconv_layers = nn.ModuleList()
        self.norm = nn.BatchNorm2d(in_features)

        for i in range(num_layers):
            f_in = in_features
            f_out = f_in
#            self.dconv_layers.append(nn.Conv2d(f_in, f_out, kernel_size=7, padding=3, bias=BIAS))
            self.dconv_layers.append(nn.ConvTranspose2d(f_in, f_out, kernel_size=7,
                                                        padding=3 * DILATION, bias=BIAS, dilation=DILATION))

    def forward(self, x, pool_indices):

        x = F.max_unpool2d(x, pool_indices, kernel_size=2, stride=2)
        x_res = x
        for i in range(self.num_layers):
            x = self.dconv_layers[i](x)
            x = ACT(x)
        x = x + x_res
        x = self.norm(x)
        return x


class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.gasf_layer = gf.GASF.apply
        self.gasf_inverse_layer = gf.InverseGASF()

        self.conv_in = nn.Conv2d(2, CHANNELS, kernel_size=1, bias=BIAS, padding=0)
        self.conv_out = nn.Conv2d(CHANNELS, 2, kernel_size=1, bias=BIAS, padding=0)

        self.se_in = SeBlock2d(CHANNELS, 8, ratio=SE_RATIO)

        self.res_blocks_in = nn.ModuleList()
        for i in range(RESIDUAL_BLOCKS):
            self.res_blocks_in.append(ResidualBlock(CHANNELS, RES_DEPTH))
        self.res_blocks_out = nn.ModuleList()
        for i in range(RESIDUAL_BLOCKS):
            self.res_blocks_out.append(DeResidualBlock(CHANNELS, RES_DEPTH))

        self.blur = nn.Conv2d(1, 1, kernel_size=5, stride=4, bias=False)
        self.blur.weight[:, :, :, :] = 1 / 5
        self.blur.weight.detach_()

        self.skip_dropouts = nn.ModuleList(nn.Dropout2d(p=DROPOUT) for _ in range(RESIDUAL_BLOCKS))
#        print(self.blur.weight)
#        print(self.blur.weight)
#        for param in self.blur.parameters():
#            param.requires_grad = False


#        self.fc_1 = nn.Linear(32 * 64**2, HIDDEN * 2)
#        self.fc_2 = nn.Linear(HIDDEN * 2, HIDDEN)
#        self.fc_3 = nn.Linear(HIDDEN, HIDDEN * 2)
#        self.fc_4 = nn.Linear(HIDDEN * 2, 32 * 64**2)

        self.apply(weights_init_)

    def forward(self, x_in):
        #        gasf, scale_min, scale_max = self.gasf_layer(x_in)
        gasf, gadf, scale_min, scale_max = self.gasf_layer(x_in)

#        x = torch.clone(gasf)
        x = torch.cat((gasf, gadf), axis=1)


#        block = torch.randint(BLOCK_MIN, BLOCK_MAX, (BATCH_SIZE,))
#        for i in range(BATCH_SIZE):
#            gasf[i, 0, block[i]:, :] = 0
#            gasf[i, 0, :, block[i]:] = 0


#        print(f'x0={x.shape}')
        x = x.reshape((BATCH_SIZE, 2, 128**2))
#        print(f'x1={x.shape}')

        mu_1 = torch.mean(x[:, 0], dim=-1)
        mu_2 = torch.mean(x[:, 1], dim=-1)
        std_1 = torch.std(x[:, 0], dim=-1)
        std_2 = torch.std(x[:, 1], dim=-1)

#        print(f'mu={mu.shape}')
#        print(f'std={std.shape}')

        for i in range(BATCH_SIZE):
            x[i, 0] = x[i, 0] - mu_1[i]
            x[i, 0] = x[i, 0] / std_1[i]
            x[i, 1] = x[i, 1] - mu_2[i]
            x[i, 1] = x[i, 1] / std_2[i]

        x = x.reshape((BATCH_SIZE, 2, 128, 128))

#        print(x.shape)

#        block = torch.randint(BLOCK_MIN, BLOCK_MAX, (1,))
        block = BLOCK_MIN
        for i in range(BATCH_SIZE):
            #            x[i, 0, block:, :] = torch.randn_like(x[i, 0, block[i]:, :]) * 1e-3
            #            x[i, 0, :, block:] = torch.randn_like(x[i, 0, :, block[i]:]) * 1e-3
            x[i, 0, block:, :] = 0
            x[i, 0, :, block:] = 0
            x[i, 1, block:, :] = 0
            x[i, 1, :, block:] = 0

        x = self.conv_in(x)
        x = F.relu(x)

        x_se = self.se_in(x)
        print(x_se.shape)

        pool_indices = list()
        skip = list()
        for i in range(RESIDUAL_BLOCKS):
            #            s1 = x.shape
            x, idx = self.res_blocks_in[i](x)
#            print(f'{i} : {s1} -> {x.shape}')
            pool_indices.insert(0, idx)
            x_skip = x
            skip.append(x_skip)

#        print(f'pre hiddne={x.shape}')
#        x = torch.flatten(x, start_dim=1)

#        x = self.fc_1(x)
#        x = ACT(x)
#        x = self.fc_2(x)

#        x = torch.zeros_like(x)

#        x = self.fc_3(x)
#        x = ACT(x)
#        x = self.fc_4(x)

#        x = x.reshape((BATCH_SIZE, 32, 64, 64))

        for i in range(RESIDUAL_BLOCKS):
            x = x + self.skip_dropouts[i](skip[-i - 1])
            x = self.res_blocks_out[i](x, pool_indices[i])

        x = self.conv_out(x)

        for i in range(BATCH_SIZE):
            x[i, 0] = x[i, 0] * std_1[i]
            x[i, 0] = x[i, 0] + mu_1[i]
            x[i, 1] = x[i, 1] * std_2[i]
            x[i, 1] = x[i, 1] + mu_2[i]

#        x_rcst = self.gasf_inverse_layer(x, scale_min, scale_max)
        x_rcst = self.gasf_inverse_layer(x[:, :1], scale_min, scale_max)

        return x, gasf, gadf, scale_min, scale_max, x_rcst, block


train_losses = list()
running_train_loss = None
running_train_loss_hist = list()


def train(model, loader: DataLoader, optimizer, epoch):

    global running_train_loss, running_train_loss_hist

    loader.dataset.set_mode('train')

    model.train()
    avg_loss = 0

    for batch_idx, (labels, ticks, mask) in enumerate(loader):

        optimizer.zero_grad()
        ticks = ticks.numpy()
        price = ticks[:, :1024, 1:2].reshape(BATCH_SIZE, 1, 1024)
#        price = price.reshape(BATCH_SIZE, 1, 513)
        ts = ticks[:, :1024, 0:1].reshape(BATCH_SIZE, 1, 1024)

        xt = np.concatenate((ts, price), axis=-2)

        xt_c = paa.paa_compress(xt, 128)

        x = np.diff(np.log(xt_c[:, 1:2, :]), axis=-1).astype(float)
        x = np.concatenate((np.zeros((BATCH_SIZE, 1, 1)), x), axis=-1)
        x = torch.from_numpy(x).to(DEVICE).float()

#        print(x.shape)

#        for i in range(BATCH_SIZE):
#            price[i, :, :] = price[i, :, :] / price[i, 0, 0]

#        print(price.shape)

#        log_returns = np.diff(np.log(price), axis=-1)
#        log_returns = np.diff(price, axis=-1)
#        x = torch.from_numpy(log_returns).to(DEVICE)

#        print(f'x={x.shape}')
#        x = torch.from_numpy(np.log(price[:, :, :512])).to(DEVICE)

        y, gasf, gadf, scale_min, scale_max, x_rcst, _ = model(x)

#        m = torch.zeros((1, 1, 512, 512)).to(DEVICE)
#        m[0, 0, 249:, :] = 1.0
#        m[0, 0, :, 249:] = 1.0

        y_down_1 = model.blur(y[:, 0:1])
        y_down_2 = model.blur(y[:, 1:2])
        gasf_down = model.blur(gasf)
        gadf_down = model.blur(gadf)
#        print(y_down_1.shape)
#        exit()

        loss_1 = F.mse_loss(y_down_1, gasf_down) + F.mse_loss(y_down_2, gadf_down)
#        loss_1 = F.mse_loss(y[:, :1], gasf) + F.mse_loss(y[:, 1:2], gadf)
#        loss_1 = torch.mean(F.mse_loss(y, gram, reduction='none') * m)
#        loss_2 = F.mse_loss(x_rcst, x[:, 0, :])
#        loss_3 = F.mse_loss(torch.cumsum(x_rcst, dim=-1), torch.cumsum(x[:, 0, :], dim=-1))

        loss = loss_1 * LOSS_1

        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= BATCH_SIZE / len(loader.dataset)
    train_losses.append(avg_loss)

    if epoch == 2:
        running_train_loss = avg_loss
        running_train_loss_hist.append(running_train_loss)
    elif epoch > 2:
        running_train_loss = (1 - 0.05) * running_train_loss + 0.05 * avg_loss
        running_train_loss_hist.append(running_train_loss)

    print(f'Train loss: {avg_loss}')


validate_losses = list()

plt_gasf = None
plt_gadf = None
plt_gasf_hat = None
plt_gadf_hat = None
plt_x_rcst = None
plt_x = None
plt_b = None

running_valid_loss = None
running_valid_loss_hist = list()


def validate(model, loader, epoch):

    global plt_gadf, plt_gasf, plt_gadf_hat, plt_gasf_hat, plt_x_rcst, plt_x, running_valid_loss, running_valid_loss_hist, plt_b

    loader.dataset.set_mode('validate')
    model.eval()
    validate_loss = 0

    with torch.no_grad():
        for batch_idx, (labels, ticks, mask) in enumerate(loader):
            ticks = ticks.numpy()

            price = ticks[:, :1024, 1:2].reshape(BATCH_SIZE, 1, 1024)
            ts = ticks[:, :1024, 0:1].reshape(BATCH_SIZE, 1, 1024)
            xt = np.concatenate((ts, price), axis=-2)
            xt_c = paa.paa_compress(xt, 128)
            x = np.diff(np.log(xt_c[:, 1:2, :]), axis=-1)
            x = np.concatenate((np.zeros((BATCH_SIZE, 1, 1)), x), axis=-1)
            x = torch.from_numpy(x).to(DEVICE).float()

            y, gasf, gadf, scale_min, scale_max, x_rcst, block = model(x)

            y_down_1 = model.blur(y[:, 0:1])
            y_down_2 = model.blur(y[:, 1:2])
            gasf_down = model.blur(gasf)
            gadf_down = model.blur(gadf)

            loss_1 = F.mse_loss(y_down_1, gasf_down) + F.mse_loss(y_down_2, gadf_down)

#            y, gram, scale_min, scale_max, x_rcst, block = model(x)

#            loss_1 = F.mse_loss(y, gram)
#            m = torch.zeros((1, 1, 512, 512)).to(DEVICE)
#            m[0, 0, 249:, :] = 1.0
#            m[0, 0, :, 249:] = 1.0

#            loss_1 = torch.mean(F.mse_loss(y, gram, reduction='none') * m)
#            loss_1 = F.mse_loss(y[:, :1], gasf) + F.mse_loss(y[:, 1:2], gadf)
#            loss_1 = F.mse_loss(y, gram)
#            loss_2 = F.mse_loss(x_rcst, x[:, 0, :])
#            loss_3 = F.mse_loss(torch.cumsum(x_rcst, dim=-1), torch.cumsum(x[:, 0, :], dim=-1))

#            loss_3 = F.mse_loss(torch.cumsum(x_rcst, dim=-1), torch.cumsum(x[:, 0, :], dim=-1))
            loss = loss_1 * LOSS_1

            validate_loss += loss

            if batch_idx == 0:
                plt_b = block
                plt_gasf = y[0, 0].cpu().numpy()
                plt_gadf = y[0, 1].cpu().numpy()
                plt_gasf_hat = gasf[0, 0].cpu().numpy()
                plt_gadf_hat = gadf[0, 0].cpu().numpy()
#                plt_gram_hat = np.concatenate(
#                    (gasf[0, 0].cpu().numpy().astype(float), gadf[0, 0].cpu().numpy()), axis=-1)
#                plt_gram = y[0, :2].cpu().numpy().astype(float)
                plt_x_rcst = x_rcst[0, :].cpu().numpy().astype(float)
                plt_x = x[0, 0, :].cpu().numpy().astype(float)
#                print(plt_x.shape)

    validate_loss *= BATCH_SIZE / len(loader.dataset)
    validate_losses.append(validate_loss)

    if epoch == 2:
        running_valid_loss = validate_loss
        running_valid_loss_hist.append(running_valid_loss)
    elif epoch > 2:
        running_valid_loss = (1 - 0.05) * running_valid_loss + 0.05 * validate_loss
        running_valid_loss_hist.append(running_valid_loss)

    print(f'Validate loss: {validate_loss}')


loader = DataLoader(dataset=TickExamples(num_examples=NUM_EXAMPLES),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True)
model = Model()
model = model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)


def main():

    for epoch in range(EPOCHS):

        train(model, loader, optimizer, epoch)
        validate(model, loader, epoch)

        rect1 = patches.Rectangle((0, 0), plt_b, plt_b, linewidth=1, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((0, 0), plt_b, plt_b, linewidth=1, edgecolor='r', facecolor='none')
        rect3 = patches.Rectangle((0, 0), plt_b, plt_b, linewidth=1, edgecolor='r', facecolor='none')
        rect4 = patches.Rectangle((0, 0), plt_b, plt_b, linewidth=1, edgecolor='r', facecolor='none')

        ax1 = plt.subplot(421)
#        vmin = np.min(plt_gram_hat.min(), plt_gram.min())
#        vmax = np.max(plt_gram_hat.max(), plt_gram.max())
        ax1.imshow(plt_gadf_hat)
        ax1.add_patch(rect1)

        ax2 = plt.subplot(422)
        ax2.imshow(plt_gadf)
        ax2.add_patch(rect2)

        ax3 = plt.subplot(423)
        ax3.imshow(plt_gasf_hat)
        ax3.add_patch(rect3)

        ax4 = plt.subplot(424)
        ax4.imshow(plt_gasf)
        ax4.add_patch(rect4)

        ax5 = plt.subplot(425)
        ax5.plot(train_losses[-100:])
        ax5.plot(validate_losses[-100:])

        ax6 = plt.subplot(426)
        ax6.plot(running_train_loss_hist)
        ax6.plot(running_valid_loss_hist)

        ax7 = plt.subplot(427)
        ax7.plot(plt_x)
        ax7.plot(plt_x_rcst)

        ax8 = plt.subplot(428)

        ax8.plot(np.exp(np.cumsum(plt_x)))
        ax8.plot(np.exp(np.cumsum(plt_x_rcst)))

        plt.show()


if __name__ == '__main__':
    main()
