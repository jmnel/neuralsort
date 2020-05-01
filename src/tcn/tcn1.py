from pprint import pprint
from random import shuffle, randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import scipy.stats as stats

from datasets.gan_examples import GanExamples

torch.manual_seed(0)

batch_size = 100

# Get CUDA if it is available and set as device.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class TcnModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = 1
        self.blocks = 1
        self.dilation_channels = 32
        self.residual_channels = 32
        self.skip_channels = 256
        self.end_channels = 256
        self.classes = 256
        self.output_length = 32,
        self.kernel_size = 2,

        self.input_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=False)

        self.tconv1 = nn.Conv1d(1, 1, 1, 1, 0, 1)
#        self.conv_in = nn.Conv1d(1, 1, 1, 1, 0, 1)

    def forward(self, x):

        print(f'in: {x.shape}')
        x[0, 0, :] = torch.arange(0, 4096)

        x = self.tconv1(x)

        print(f'res1: {x.shape}')

        pass


class Tcn:

    def __init__(self, epochs):

        self.epochs = epochs

        self.model = TcnModel()
        self.loader = DataLoader(
            GanExamples(),
            batch_size=1,
            shuffle=True,
            drop_last=True)

    def train(self, epoch):

        for idx, (x, ts) in enumerate(self.loader):
            y = self.model(x.view((1, 1, 4096)))
            print(y)
            exit()

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)


EPOCHS = 2000
tcn = Tcn(EPOCHS)
tcn.run()
