import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # Add zero padding for reshaping.
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # Reshape according to dilation.
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class ConstantPad1d(Function):

    def __init__(self,
                 target_size,
                 dimension=0,
                 value=0,
                 pad_start=False):

        super().__init__()
        self.target_size = target_size
        self.dimension = dimension
        self.value = value
        self.pad_start = pad_start

#    @staticmethod
    def forward(self, x_input):
        self.num_pad = self.target_size - x_input.size(self.dimension)

        assert self.num_pad >= 0, 'target size has to be greater than input size'

        self.input_size = x_input.size()

        print(f'input size: {x_input.size()}')

        size = list(x_input.size())
        size[self.dimension] = self.target_size
        x_output = x_input.new(*tuple(size)).fill_(self.value)
        c_output = x_output

        # Crop the output.
        if self.pad_start:
            c_output = c_output.narrow(self.dimension,
                                       self.num_pad,
                                       c_output.size(self.dimension) - self.num_pad)
        else:
            c_output = c_output.narrow(self.dimension,
                                       0,
                                       c_output.size(self.dimension) - self.num_pad)

        c_output.copy_(x_input)

        print(f'output size: {x_output.shape}')
        return x_output

#    @staticmethod
    def backward(self, grad_output):

        print(f'grad output shape {grad_output.shape}')
        grad_input = grad_output.new(*self.input_size).zero_()
        cg_output = grad_output

        # Crop the gradient output.
        if self.pad_start:
            cg_output = cg_output.narrow(self.dimension,
                                         self.num_pad,
                                         cg_output.size(self.dimension) - self.num_pad)

        else:
            cg_output = cg_output.narrow(self.dimension,
                                         0,
                                         cg_output.size(self.dimension) - self.num_pad)

        grad_input.copy_(cg_output)

        print(f'grad input shape {grad_input.shape}')

        return grad_input


def constant_pad_1d(x_input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):

    return ConstantPad1d(target_size, dimension, value, pad_start)(x_input)


class DilatedQueue:

    def __init__(self,
                 max_length,
                 data=None,
                 dilation=1,
                 num_deq=1,
                 num_channels=1,
                 dtype=torch.FloatTensor):

        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype

        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, x_input):

        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):

        start = self.out_pos - ((num_deq - 1) * dilation)

        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos %
                           dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)

        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(
            self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0
