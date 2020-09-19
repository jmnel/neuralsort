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
BATCH_SIZE = 1
NUM_EXAMPLES = 1


class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.gasf_layer = gf.GASF(DEVICE)
        self.gasf_inverse_layer = gf.InverseGASF(DEVICE)

    def forward(self, x_in):
        gasf, scale_min, scale_max = self.gasf_layer(x_in)

        x = torch.clone(gasf)
        x_rcst = self.gasf_inverse_layer(x, scale_min, scale_max)

        return x, gasf, scale_min, scale_max, x_rcst


loader = DataLoader(dataset=TickExamples(num_examples=NUM_EXAMPLES),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True)
model = Model()
model = model.to(DEVICE)

labels, ticks, mask = next(iter(loader))
ticks = ticks.numpy()
price = ticks[:, :513, 1:2]

price = price.reshape(BATCH_SIZE, 1, 513)

log_returns = np.diff(np.log(price), axis=-1)
x = torch.from_numpy(log_returns).to(DEVICE)

y, gram, scale_min, scale_max, x_rcst = model(x)
x = x.cpu().numpy()
gram = gram.cpu().numpy()
x_rcst = x_rcst.cpu().numpy()[0]

fig, axes = plt.subplots(2, 1)

axes[0].plot(x[0, 0])
axes[1].plot(x_rcst)
#axes[1].imshow(gram[0, 0])

plt.show()
