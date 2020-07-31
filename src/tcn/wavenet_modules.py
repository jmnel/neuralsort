import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x,
           dilation,
           init_dilation=1,
           pad_start=True):

    [n, c, t] = x.shape
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # Add zero padding before reshape operation.
    t_new = int(np.ceil(t / dilation_factor) * dilation_factor)

    if t_new != t:
        t = t_new
        x = const_pad_1d(x,
                         t_new,
                         dim=2,
                         pad_start=pad_start)

    t_old = int(round(t / dilation_factor))
    n_old = int(round(n * dilation_factor))

    t = math.ceil(t * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # Reshape according to dilation.
    x = x.permute(1, 2, 0).contiguous()  # (n, c, t) -> (c, t, n)
    x = x.view(c, t, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, t, n) -> (n, c, t)

    return x


class ConstPad1d(Function):

    @staticmethod
    def forward(ctx,
                x_input,
                target_size,
                dim=0,
                value=0,
                pad_start=False):

        num_pad = target_size - x_input.shape[dim]

        assert num_pad >= 0, 'resulting (target) size must be greater than input size'

        x_size = x_input.shape

        # Save arguments to context.
        ctx.x_size = x_size
        ctx.num_pad = num_pad
        ctx.dim = dim
        ctx.pad_start = pad_start

        size = list(x_size)
        size[dim] = target_size

        x_output = x_input.new(*tuple(size)).fill_(value)
        c_output = x_output

        # Crop the output.
        if pad_start:
            c_output = c_output.narrow(dim,
                                       num_pad,
                                       c_output.shape[dim] - num_pad)

        else:
            c_output = c_output.narrow(dim,
                                       0,
                                       c_output.shape[dim] - num_pad)

        c_output.copy_(x_input)
        return x_output

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.new(*ctx.x_size).zero_()
        cg_output = grad_output

        # Crop gradient output.
        if ctx.pad_start:
            cg_output = cg_output.narrow(ctx.dim,
                                         ctx.num_pad,
                                         cg_output.shape[ctx.dim] - ctx.num_pad)

        else:
            cg_output = cg_output.narrow(ctx.dim,
                                         0,
                                         cg_output.shape[ctx.dim] - ctx.num_pad)

        grad_input.copy_(cg_output)

        # Only first input is a variable along which to calculate gradient.
        return grad_input, None, None, None, None


def const_pad_1d(x_input,
                 target_size,
                 dim=0,
                 value=0,
                 pad_start=False):

    # Apply function object to input.
    return ConstPad1d.apply(x_input, target_size, dim, value, pad_start)


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
        self.max_length = max_length
        self.data = data
        self.dilation = dilation
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dtype = dtype

        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, x_input):
        print(f'in_pos: {self.in_pos}')

        print(f'self data: {self.data.shape}')
        print(f'max len: {self.max_length}')

        print(f'data set')
        print(f'lhs: {self.data[:, self.in_pos].shape}')
        print(f'rhs: {x_input.shape}')

        self.data[:, self.in_pos] = x_input

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
        self.in_pos, self.out_pos = 0, 0
