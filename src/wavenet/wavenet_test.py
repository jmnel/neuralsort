import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pprint import pprint

from wavenet_modules2 import *


# print(y.shape)

#from wavenet_nn import WaveNetNN

# wn = WaveNetNN(layers=10,
#               blocks=4,
#               dilation_channels=32,
#               residual_channels=32,
#               skip_channels=256,
#               end_channels=256,
#               input_channels=1,
#               output_channels=1,
#               kernel_size=1)

#x = torch.FloatTensor(1, 1, 100)

# wn.eval()

#y = wn(x)
