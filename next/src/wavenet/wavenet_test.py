import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from wavenet_model import WaveNetModel

model = WaveNetModel()

x = torch.FloatTensor(1, 256, 100)

model.eval()

y = model(x)
