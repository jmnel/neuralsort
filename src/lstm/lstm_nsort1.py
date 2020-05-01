from pathlib import Path
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

num_layers = 1
hidden_size = 100
train_size = 6000
validate_size = 2000
test_size = 2000

train_batch_size = 1
validate_batch_size = 100
test_batch_size = 100

epochs = 2000
forecast_window = 100


class LstmModel1(nn.Module):

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

        self.hidden_cell = (torch.zeros(num_layers, 1, hidden_size).to(device),
                            torch.zeros(num_layers, 1, hidden_size).to(device))

    def forward(self, x):

        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        lstm_out_last = lstm_out[:, -1, :]

        print(lstm_out.shape)
        exit()

        y = self.linear(lstm_out_last)

#        print(y.shape)

#        y = torch.stack(tuple(y[(i * x.shape[0]):(i + 1) * x.shape[0], :]
#                              for i in range(num_stocks)), dim=1)

#        print(y)
#        exit()

        return y


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0

    for batch_idx, (x, y_true, _) in enumerate(train_loader):

        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)

#        y_true = y_true.reshape(train_batch_size, num_stocks, 5)
        y_true = y_true[:, 3::5].reshape(train_batch_size, num_stocks, 1)

        model.hidden_cell = (torch.zeros(model.num_layers,
                                         model.num_stocks * train_batch_size,
                                         model.hidden_size).to(device),
                             torch.zeros(model.num_layers,
                                         model.num_stocks * train_batch_size,
                                         model.hidden_size).to(device))

        x = torch.cat(tuple(x[:, :, i, :]
                            for i in range(x.shape[2])),
                      dim=0)

        y = model(x)

        y = torch.stack(tuple(y[(i * train_batch_size):(i + 1) * train_batch_size, :]
                              for i in range(num_stocks)), dim=1)

        p_true = compute_permu_matrix(y_true, 1e-10)
        p_hat = compute_permu_matrix(y, 5)

        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20),
                          dim=1).mean()

        loss.backward()
        optimizer.step()

        avg_loss += loss

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
            x = torch.cat(tuple(x[:, :, i, :]
                                for i in range(x.shape[2])),
                          dim=0)

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
               use_cache=True,
               forecast_window=forecast_window),
    batch_size=train_batch_size,
    shuffle=True)
print('done')

# Initialize validation data loader.
print('initializing validation loader...')
validate_loader = DataLoader(
    WindowNorm(num_samples=validate_size,
               mode='validate',
               use_cache=True,
               forecast_window=forecast_window),
    batch_size=validate_batch_size,
    shuffle=True)
print('done')

# Initialize test data loader.
print('initializing test loader...')
test_loader = DataLoader(
    WindowNorm(num_samples=test_size,
               mode='test',
               use_cache=True,
               forecast_window=forecast_window),
    batch_size=test_batch_size,
    shuffle=True)
print('done')

num_stocks = train_loader.dataset.num_stocks
num_attributes = train_loader.dataset.num_attributes

model = LstmModel1(num_layers=num_layers,
                   hidden_size=hidden_size,
                   num_stocks=num_stocks,
                   num_attributes=num_attributes,
                   device=device).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    validate(model, device, validate_loader)
