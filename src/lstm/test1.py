import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 100, 1000)
x = np.sin(t * 0.1 * np.pi) * np.mod(t, 20) + np.random.randn(1000) * 2

test_size = 200
t_train = t[:-test_size]
x_train = x[:-test_size]

window = 10
data_train = list()

for i in range(80):
    seq = torch.FloatTensor(x_train[10 * i:10 * (i + 1) - 1])
    label = torch.FloatTensor([x_train[10 * (i + 1) - 1]])
    seq_t = torch.FloatTensor(t_train[10 * i:10 * (i + 1) - 1])
    label_t = t_train[10 * (i + 1) - 1]

    data_train.append((seq, label, seq_t, label_t))

# for sample in data_train:
#    seq, label, seq_t, label_t = sample

#    plt.scatter(seq_t, seq, c='blue', s=4)

#    plt.scatter([label_t], [label], c='red', s=4)


class LSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=100,
                            num_layers=4)

        self.linear = nn.Linear(in_features=100, out_features=1)
        self.hidden_cell = (torch.zeros(4, 1, 100),
                            torch.zeros(4, 1, 100))

    def forward(self, seq_x):

        lstm_out, self.hidden_cell = self.lstm(seq_x.view(len(seq_x), 1, -1),
                                               self.hidden_cell)

        pred = self.linear(lstm_out.view(len(seq_x), -1))

        return pred[-1]


model = LSTM()
model = model.to('cuda')
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):

    for foo in data_train:
        seq, label, _, _ = foo
        optimizer.zero_grad()

        seq = seq.to('cuda')
        label = label.to('cuda')

        model.hidden_cell = (torch.zeros(4, 1, 100).to('cuda'),
                             torch.zeros(4, 1, 100).to('cuda'))

        pred = model(seq)

        l = loss(pred, label)
        l.backward()
        optimizer.step()

    print(f'epoch: {i} loss: {l.item()}')


# model = model.to('cpu')

for sample in data_train:
    seq, label, seq_t, label_t = sample
    seq = seq.to('cuda')

    with torch.no_grad():
        pred = model(seq)

        pred = pred.to('cpu')

        plt.plot(seq_t, seq.cpu(), linewidth=0.4, c='blue')
        plt.scatter(label_t, pred.cpu(), s=2, c='red')
#    print(x)
#    print(pred)


# plt.plot(t, data_x)
plt.show()
