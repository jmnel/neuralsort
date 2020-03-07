import torch
import torch.nn as nn


a = torch.arange(50).reshape((1, 5, 10))

print(a)

print(len(a))
b = a.view(len(a), 1, -1)

print(b)
