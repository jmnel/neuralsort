from pathlib import Path
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from datasets.window_norm import WindowNorm
from nsort import compute_permu_matrix, prop_correct, prop_any_correct

torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

top_k = 3
num_layers = 1
hidden_size = 120
train_size = 600
validate_size = 200
test_size = 200

train_batch_size = 20
validate_batch_size = 20
test_batch_size = 20

epochs = 2000
forecast_window = 100


class LstmModel2(nn.Module):

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_stocks,
                 num_attributes,
                 device):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_stocks = num_stocks
        self.num_attributes = num_attributes

        self.lstm = nn.LSTM(input_size=num_attributes,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=1)

        self.linear2 = nn.Linear(in_features=self.num_stocks * self.num_stocks,
                                 out_features=256)
        self.linear3 = nn.Linear(in_features=256,
                                 out_features=self.num_stocks)

#        self.hidden_cell = (torch.zeros(num_stocks, num_layers, 1, hidden_size).to(device),
#                            torch.zeros(num_stocks, num_layers, 1, hidden_size).to(device))

    def forward(self, x):

        #        print(f'x shape is {x.shape}')

        self.hidden_cell = list(
            (torch.zeros(
                self.num_layers,
                x.shape[0],
                self.hidden_size).to(device),
             torch.zeros(
                self.num_layers,
                x.shape[0],
                self.hidden_size).to(device))
            for _ in range(self.num_stocks))

        lstm_out = torch.zeros(
            (x.shape[0], self.num_stocks, x.shape[1], self.hidden_size))

        for i in range(num_stocks):
            #        lstm_out, self.hidden_cell = self.lstm(x[:, :, 0, :], self.hidden_cell)
            lstm_out[:, i, :, :], self.hidden_cell[i] = self.lstm(
                x[:, :, i, :], self.hidden_cell[i])

        lstm_out_last = lstm_out[:, :, -1, :].to(device)

        score = torch.zeros((x.shape[0], self.num_stocks, 1)).to(device)

#        for i in range(self.num_stocks):

        for idx in range(x.shape[0]):
            for jdx in range(self.num_stocks):
                score[idx, jdx, 0] = self.linear(lstm_out_last[idx, jdx, :])

#        print(score.shape)

        p_hat = compute_permu_matrix(score, tau=5)

        foo = p_hat.flatten(start_dim=1)

        z = self.linear2(foo)
        z = F.relu(z)

        z = self.linear3(z)
        z = F.softmax(z)

        return z

#        print(z.shape)

#        exit()

#        score = torch.FloatTensor(self.linear(
#            list(lstm_out_last[:, 0, :]) for i in range(self.num_stocks)))

#        score[0, 0] = self.linear(lstm_out_last[0, 0, :])
#        score[1, 0] = self.linear(lstm_out_last[1, 0, :])
#        print(score.shape)

#        print(lstm_out.shape)
#        print(lstm_out_last.shape)


#        lstm_out, self.hidden_cell = self.lstm(
#                x[i],
#        for i in range(num_stocks):
#            print(self.hidden_cell.shape)
#            lstm_out, (self.hidden_cell[0][i], self.hidden_cell[1][i]) = self.lstm(
#                x[i], (self.hidden_cell[0][i], self.hidden_cell[1][i])
#            print(lstm_out.shape)
#            exit()
#            lstm_out_last = lstm_out[:, -1, :]

#        y = self.linear(lstm_out_last)

#        print(y.shape)

#        y = torch.stack(tuple(y[(i * x.shape[0]):(i + 1) * x.shape[0], :]
#                              for i in range(num_stocks)), dim=1)

#        print(y)
#        exit()

#        return y


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0

    for batch_idx, (x, y_true, actual) in enumerate(train_loader):

        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)
        actual = actual.to(device)

        open_true = actual[:, :, 0]
        close_true = actual[:, :, 3]
        rel_return_true = (close_true - open_true) / open_true

        rank_true = torch.argsort(rel_return_true)

#        print(rank_true)
        top_k_true = rank_true < top_k
        top_k_true = top_k_true.float()

        y = model(x)

        loss = F.binary_cross_entropy(y, top_k_true)

        loss.backward()
        optimizer.step()

        avg_loss += loss


#        exit()

#        y_true = y_true.reshape(train_batch_size, num_stocks, 5)
#        y_true = y_true[:, 3::5].reshape(train_batch_size, num_stocks, 1)

#        model.hidden_cell = (torch.zeros(model.num_layers,
#                                         model.num_stocks * train_batch_size,
#                                         model.hidden_size).to(device),
#                             torch.zeros(model.num_layers,
#                                         model.num_stocks * train_batch_size,
#                                         model.hidden_size).to(device))


#        x = torch.cat(tuple(x[:, :, i, :]
#                            for i in range(x.shape[2])),
#                      dim=0)

#        print(y.shape)


#        y = torch.stack(tuple(y[(i * train_batch_size):(i + 1) * train_batch_size, :]
#                              for i in range(num_stocks)), dim=1)

#        p_true = compute_permu_matrix(y_true, 1e-10)
#        p_hat = compute_permu_matrix(y, 5)

#        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20),
#                          dim=1).mean()

    avg_loss *= train_batch_size / train_size

    print(f'Epoch {epoch}: avg. train loss: {avg_loss}')


def validate(model, device, validate_loader):

    model.eval()
    validate_loss = 0.

    correct = 0
    any_correct = 0.0

    with torch.no_grad():

        for x, y_true, _ in validate_loader:

            x, y_true = x.to(device), y_true.to(device)

            model.hidden_cell = (torch.zeros(model.num_layers,
                                             model.num_stocks * validate_batch_size,
                                             model.hidden_size).to(device),
                                 torch.zeros(model.num_layers,
                                             model.num_stocks * validate_batch_size,
                                             model.hidden_size).to(device))

#            y_true = y_true.reshape(validate_batch_size, num_stocks, 1)
            y_true = y_true[:, 3::5].reshape(test_batch_size, num_stocks, 1)
#            x = torch.cat(tuple(x[:, :, i, :]
#                                for i in range(x.shape[2])),
#                          dim=0)

            y = model(x)

            y = torch.stack(tuple(y[(i * validate_batch_size):(i + 1) * validate_batch_size, :]
                                  for i in range(num_stocks)), dim=1)

            p_true = compute_permu_matrix(y_true, 1e-10)
            p_hat = compute_permu_matrix(y, 5)

            validate_loss += -torch.sum(p_true * torch.log(p_hat + 1e-20),
                                        dim=1).mean()

            correct += prop_correct(p_true, p_hat)
            any_correct += prop_any_correct(p_true, p_hat)

    validate_loss *= validate_batch_size / validate_size

    print('\nValidation avg. loss: {:.4f}'.format(validate_loss))
    print('  all correct: {:.0f} / {:.0f} = {:.1f}%'.format(
        correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset)))
    print('  any correct: {:.1f}%'.format(
        100. * any_correct * validate_batch_size / len(validate_loader.dataset)))
    print()


# Initialize train data loader.
print('initializing train loader...')
train_loader = DataLoader(
    WindowNorm(num_samples=train_size,
               mode='train',
               use_cache=True),
    batch_size=train_batch_size,
    shuffle=True)
print('done')

# Initialize validation data loader.
print('initializing validation loader...')
validate_loader = DataLoader(
    WindowNorm(num_samples=validate_size,
               mode='validate',
               use_cache=True),
    batch_size=validate_batch_size,
    shuffle=True)
print('done')

# Initialize test data loader.
print('initializing test loader...')
test_loader = DataLoader(
    WindowNorm(num_samples=test_size,
               mode='test',
               use_cache=True),
    batch_size=test_batch_size,
    shuffle=True)
print('done')

num_stocks = train_loader.dataset.num_stocks
num_attributes = train_loader.dataset.num_attributes

model = LstmModel2(num_layers=num_layers,
                   hidden_size=hidden_size,
                   num_stocks=num_stocks,
                   num_attributes=num_attributes,
                   device=device).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
#    validate(model, device, validate_loader)
