import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmModel(nn.Module):

    def __init__(self,
                 num_layers=1,
                 hidden_size=100,
                 num_seqences=5,
                 device='cpu'):

        super().__init__()

        self.lstm = nn.LSTM(input_size=num_seqences,
                            hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=num_seqences)
        self.hidden_cell = (torch.zeros(num_layers, 2, hidden_size).to(device),
                            torch.zeros(num_layers, 2, hidden_size).to(device))

    def forward(self, x):

        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        y = self.linear(lstm_out)
#        last = y[:, -1, :]

        return y[:, -1, :]
