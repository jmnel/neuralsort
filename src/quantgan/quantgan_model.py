from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavenet.wavenet_model import WaveNetModel
from quantgan_dataset import QuantGanDataset


class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.tcn = WaveNetModel(layers=10,
                                blocks=4,
                                dilation_channels=32,
                                residual_channels=32,
                                skip_channels=32,
                                end_channels=32,
                                #                                output_length=4096,
                                input_channels=1,
                                output_channels=2)

        print(self.tcn.receptive_field)

    def forward(self,
                x_input):

        x = self.tcn(x_input)

        pprint(x)

        print(x.shape)
        pass

#    def generate(self):

#        self.tcn.generate(12)


m = Generator()
#q = m.generate()

z = torch.randn((2, 1, 4093))

data = QuantGanDataset()
x, meta, symbols = next(iter(data))

x = x.view(1, 1, 4096)

y = m(x)
