from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavenet.wavenet_model import WaveNetModel
from quantgan_dataset import QuantGanDataset


class InnovationNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear1 = nn.Linear(1, 128)
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 1)

    def forward(self, x_input):

        x = self.linear1(x_input)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)

        return x


class GeneratorNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.drift_tcn = WaveNetModel(layers=10,
                                      blocks=4,
                                      dilation_channels=32,
                                      residual_channels=32,
                                      skip_channels=32,
                                      end_channels=32,
                                      input_channels=1,
                                      output_channels=1,
                                      output_length=32)

        self.volatiliy_tcn = WaveNetModel(layers=10,
                                          blocks=4,
                                          dilation_channels=32,
                                          residual_channels=32,
                                          skip_channels=32,
                                          end_channels=32,
                                          input_channels=1,
                                          output_channels=1,
                                          output_length=32)

        self.innov = Innovation()

    def forward(self,
                x_input):

        mu = self.drift_tcn(x_input[:, 0:1, :])
        vol = self.volatiliy_tcn(x_input[:, 1:2, :])
        eps = self.innov(x_input[:, 2, -1])

        y = mu + vol * eps

        return y

#        print(x.shape)
#        pass

    def generate(self):

        self.drift_tcn(

#    def generate(self):

#        self.tcn.generate(12)


m=Generator()
# q = m.generate()

z=torch.randn((2, 1, 4093))

data=QuantGanDataset()
x, meta, symbols=next(iter(data))

x=x.view(1, 1, 4096)

y=m(x)
