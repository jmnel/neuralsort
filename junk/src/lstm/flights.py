import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
from pprint import pprint

flight_data = sns.load_dataset('flights')
all_data = flight_data['passengers'].values.astype(float)

# all_data = np.sin(np.linspace(0, 20, len(all_data))).astype(float)

# all_data.astype(float)

test_data_size = 12

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(
    train_data.reshape(-1, 1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_window = 12


def create_inout_seq(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_seq(train_data_normalized, train_window)


class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1),
                                               self.hidden_cell)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# x = train_inout_seq
# x = x.to('cuda')

model = LSTM()

model = model.to('cuda')
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#epochs = 5

# for i in range(epochs):
#    for seq, labels in train_inout_seq:

#        seq = seq.to('cuda')
#        labels = labels.to('cuda')

#        optimizer.zero_grad()
#        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to('cuda'),
#                             torch.zeros(
#                                 1, 1, model.hidden_layer_size).to('cuda'))

#        y_pred = model(seq)

#        single_loss = loss(y_pred, labels)
#        single_loss.backward()
#        optimizer.step()

#    if i % 25 == 1:
#        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

#print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

# t_dom = list([
#    list([float(i + j * 11) for i in range(12)])
#    for j in range(12)])

# assert(len(t_dom) == len(train_inout_seq))

# x, y = train_inout_seq[0]

# xd = [scaler.inverse_transform(np.array(q[0]).reshape(-1, 1))
#      for q in train_inout_seq]

#xd = list([q[0] for q in train_inout_seq])

# for q in train_inout_seq:
#    print(q)

# exit()

#k = 4

# print(len(labels))

# plt.plot(train_data_normalized)
#plt.plot([i + 12 * k + 1 for i in range(12)], xd[k])
#plt.xticks([i * 12 for i in range(12)])
# plt.grid()

# plt.plot(xd)
# plt.show()
