import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

from wavenet_modules import *


class WaveNetNN(nn.Module):

    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 end_channels=256,
                 classes=256,
                 #                 input_channels=1,
                 #                 output_channels=2,
                 output_length=32,
                 kernel_size=2,
                 bias=False):

        super().__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.classes = classes
#        self.input_channels = input_channels
#        self.output_channels = output_channels
        self.output_length = output_length
        self.kernel_size = kernel_size

        # Build the model.
        receptive_field = 1
        init_dilation = 1

        self.dilations = list()
        self.dilated_queues = list()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # A 1x1 convolution creates input channels by projection.
#        self.start_conv = nn.Conv1d(in_channels=input_channels,
#                                    out_channels=residual_channels,
#                                    kernel_size=1,
#                                    bias=bias)
        self.start_conv = nn.Conv1d(in_channels=classes,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):

            additional_scope = kernel_size - 1
            new_dilation = 1

            for i in range(layers):
                self.dilations.append((new_dilation, init_dilation))

                # Dilated queues for fast generation.
                self.dilated_queues.append(DilatedQueue(
                    max_length=(kernel_size - 1) * new_dilation + 1,
                    num_channels=residual_channels,
                    dilation=new_dilation,
                    dtype=torch.FloatTensor))

                # Dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection.
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection.
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=1,
                                    bias=True)

#        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
#                                    out_channels=output_channels,
#                                    kernel_size=1,
#                                    bias=True)
        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, x_input, dilation_func):

        x = self.start_conv(x_input)

        print('here')
        print(x.shape)

        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            print(f'layer {i}')

            (dilation, init_dilation) = self.dilations[i]

            x_residual = dilation_func(x, dilation, init_dilation, i)

            print(f'after dilate: {x_residual.shape}')

            # Dilated convolution
            x_filter = self.filter_convs[i](x_residual)
            x_filter = torch.tanh(x_filter)
            x_gate = self.gate_convs[i](x_residual)
            x_gate = torch.sigmoid(x_gate)
            x = x_filter * x_gate

            # Parametrized skip connections
            x_s = x
            if x.shape[2] != 1:
                x_s = dilate(x, 1, init_dilation=dilation)
            x_s = self.skip_convs[i](x_s)

            try:
                x_skip = skip[:, :, -x_s.shape[2]:]
            except:
                x_skip = 0
            x_skip = x_s + x_skip

            x = self.residual_convs[i](x)
            x = x + x_residual[:, :, (self.kernel_size - 1):]

        x = F.relu(x_skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def wavenet_dilate(self, x_input, dilation, init_dilation, i):
        x = dilate(x_input, dilation, init_dilation)
        return x

    def queue_dilate(self, x_input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]

        print(f'before queue: {x_input.data[0].shape}')

        queue.enqueue(x_input.data[0])

        x = queue.dequeue(num_deq=self.kernel_size, dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, x_input):

        x = self.wavenet(x_input,
                         dilation_func=self.wavenet_dilate)

        # Reshape the output.
        [n, c, l] = x.shape

        t = self.output_length
        x = x[:, :, -t:]
        x = x.transpose(1, 2).contiguous()

        x = x.view(n * t, c)

        return x

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      #                      temperature=1.,
                      regularize=0.):

        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(
                1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # Reset queues.
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.shape[0]
        total_samples = num_given_samples + num_samples

        x_input = Variable(torch.FloatTensor(1,
                                             self.classes,
                                             1).zero_())
#        x_input = x_input.scatter_(1,
#                                   first_samples[0:1].view(1, -1, 1), 1.)

        # Fill queues with starting samples.
        for i in range(num_given_samples - 1):
            x = self.wavenet(x_input,
                             dilation_func=self.queue_dilate)

            x_input.zero_()
#            x_input = x_input.scatter_(1,
#                                       first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

        # Generate new samples.
        generated = np.array([])
#        regularizer = torch.pow(Variable(torch.arange(self.classes))
#                                - self.classes / 2., 2)
#        regularizer = regularizer.squeeze() * regularize

        for i in range(num_samples):

            print(x_input.shape)
            x = self.wavenet(x_input,
                             dilation_func=self.queue_dilate).squeeze()
            exit()


#            x -= regularizer

#            if temperature > 0:
#                x /= temperature
#                prob = F.softmax(x, dim=0)
