import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

# import matplotlib
# matplotlib.use('Qt5Cairo')
# import matplotlib.pyplot as plt

from db_connectors import SQLite3Connector
from datasets import Windowed10Dataset

torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_layers = 1
hidden_size = 100
train_size = 6400
test_size = 1600
train_batch_size = 100
test_batch_size = 100

epochs = 200
forecast_window = 100
num_stocks = 5


def bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1).to(device)
    b = bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0).to(device)
    c = bl_matmul(s, d)
    mat_p = (c - b).permute(0, 2, 1)
    mat_p = F.softmax(mat_p / tau, -1)

    return mat_p


def prop_any_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2).float()
    correct = torch.mean(eq, axis=-1)
    return torch.mean(correct)


def prop_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2)
    correct = torch.all(eq, axis=-1).float()
    return torch.sum(correct)


class LstmOhlcvModelSerial(nn.Module):

    def __init__(self,
                 num_layers=1,
                 hidden_size=100,
                 num_stocks=5,
                 num_attributes=5,
                 device='cuda'):

        super().__init__()

        self.lstm = nn.LSTM(input_size=num_attributes,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=500)

        self.linear2 = nn.Linear(in_features=500,
                                 out_features=100)

        self.linear3 = nn.Linear(in_features=100,
                                 out_features=num_stocks)

#        self.drop0 = nn.Dropout(p=0.2)
#        self.drop1 = nn.Dropout(p=0.2)

        self.hidden_cell = (torch.zeros(num_layers, 2, hidden_size).to(device),
                            torch.zeros(num_layers, 2, hidden_size).to(device))

    def forward(self, x):

        #        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        y = self.linear(lstm_out)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)

        return y[:, -1, :]


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0.

    for batch_idx, (x, y_true) in enumerate(train_loader):

        optimizer.zero_grad()

        x, y_true = x.to(device), y_true.to(device)

        model.hidden_cell = (torch.zeros(num_layers,
                                         train_batch_size * num_stocks,
                                         hidden_size).to(device),
                             torch.zeros(num_layers,
                                         train_batch_size * num_stocks,
                                         hidden_size).to(device))

        y_true = y_true[:, 3::5].reshape(train_batch_size, num_stocks, 1)

#        y = model(x).reshape(train_batch_size, num_stocks, 1)
#        print(x.shape)

        x = torch.cat(tuple(x[:, :, (i * 5):(i + 1) * 5]
                            for i in range(num_stocks)), dim=0)
#        print(x.shape)
#        exit()
        y = model(x)

#        print(y.shape)
#        y = torch.FloatTensor([y[i * 100:(i + 1) * 100]
#                               for i in range(num_stocks)])
#        y = torch.FloatTensor(list(
#            y[i * train_batch_size:(i + 1) * train_batch_size, :] for i in range(num_stocks)))
#        y = torch.FloatTensor( tuple(x[
        y = torch.stack(tuple(y[(i * train_batch_size):(i + 1) * train_batch_size, :]
                              for i in range(num_stocks)), dim=1)

        print(y.shape)
#        exit()

        loss = torch.FloatTensor([0.])
        for k in range(train_batch_size):
            p_true = compute_permu_matrix(y_true[k], 1e-10)
            p_hat = compute_permu_matrix(y[k], 5)

            loss += -torch.sum(p_true * torch.log(p_hat + 1e-20),
                               dim=1).mean()

        avg_loss += loss

        loss.backward()
        optimizer.step()

    avg_loss *= train_batch_size / train_size

    print(f'Epoch {epoch}: avg. loss: {avg_loss}')


test_loss_degrade = 0
past_test_loss = float('inf')


def test(model, device, test_loader):

    model.eval()
    test_loss = 0.

    correct = 0
    any_correct = 0.0

    with torch.no_grad():

        for x, y_true in test_loader:

            x, y_true = x.to(device), y_true.to(device)

            model.hidden_cell = (torch.zeros(num_layers,
                                             test_batch_size,
                                             hidden_size).to(device),
                                 torch.zeros(num_layers,
                                             test_batch_size,
                                             hidden_size).to(device))

            y_true = y_true[:, 3::5].reshape(test_batch_size, num_stocks, 1)
            y = model(x).reshape(test_batch_size, num_stocks, 1)

            p_true = compute_permu_matrix(y_true, 1e-10)
            p_hat = compute_permu_matrix(y, 5)

            test_loss += -torch.sum(p_true * torch.log(p_hat + 1e-20),
                                    dim=1).mean()

            correct += prop_correct(p_true, p_hat)
            any_correct += prop_any_correct(p_true, p_hat)

    test_loss *= test_batch_size / test_size

    global past_test_loss
    global test_loss_degrade
    if test_loss <= past_test_loss:
        past_test_loss = test_loss
        test_loss_degrade = 0
    else:
        test_loss_degrade += 1

    print('\nTest avg. loss: {:.4f}'.format(test_loss))
    print('  all correct: {:.0f} / {:.0f} = {:.1f}%'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('  any correct: {:.1f}%'.format(
        100. * any_correct * test_batch_size / len(test_loader.dataset)))
    print()


data_path = Path(__file__).absolute().parents[2] / 'data'

print('Setting up train data loader...')
train_loader = DataLoader(
    Windowed10Dataset(data_path=data_path,
                      num_samples=train_size,
                      forecast_window=forecast_window,
                      num_stocks=num_stocks,
                      train=True),
    batch_size=train_batch_size,
    shuffle=True)

print('Setting up test data loader...')
test_loader = DataLoader(
    Windowed10Dataset(data_path=data_path,
                      num_samples=test_size,
                      forecast_window=forecast_window,
                      num_stocks=num_stocks,
                      train=False),
    batch_size=test_batch_size,
    shuffle=True)

model = LstmOhlcvModelSerial(num_layers=num_layers, hidden_size=hidden_size,
                             num_stocks=num_stocks,
                             num_attributes=5,
                             device=device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
#    if test_loss_degrade > 9:
#        print('Test loss not improving.')
#        print('Stopping early to prevent overfitting.')
#        break
