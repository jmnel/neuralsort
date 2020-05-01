from pathlib import Path
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('Qt4agg')
import matplotlib.pyplot as plt

from datasets.window_norm import WindowNorm
from nsort import compute_permu_matrix, prop_correct, prop_any_correct

torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_stocks = 50
top_k = 3
num_layers = 1
hidden_size = 200
train_size = 6000
validate_size = 2000
test_size = 2000

train_batch_size = 200
validate_batch_size = 200
test_batch_size = 200

epochs = 2000
forecast_window = 100


def prop_top_k_correct(r, r_true, k):
    r = torch.argsort(r)
    r = r < top_k
    return torch.sum(r == r_true).item()
#    print(r)
#    print(r_true)
#    exit()
#    r = list(r[i] < k for i in range(num_stocks))

#    return sum(1 if r[i] == r_true[i] else 0 for i in range(num_stocks))


class LstmModel3(nn.Module):

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

        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(input_size=num_attributes * num_stocks,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear1_1 = nn.Linear(in_features=hidden_size * forecast_window,
                                   out_features=512)
        self.linear1_2 = nn.Linear(in_features=512,
                                   out_features=64)
#        self.linear1_3 = nn.Linear(in_features=64,
#                                   out_features=self.num_stocks)
        self.linear1_3 = nn.Linear(in_features=hidden_size * forecast_window,
                                   out_features=self.num_stocks)

#        self.linear2_1 = nn.Linear(in_features=self.num_stocks * self.num_stocks,
#                                   out_features=512)
#        self.linear2_1 = nn.Linear(in_features=self.num_stocks * self.num_stocks,
#                                   out_features=self.num_stocks)
        self.linear2_2 = nn.Linear(in_features=512,
                                   out_features=64)
#        self.linear2_3 = nn.Linear(in_features=64,
#                                   out_features=self.num_stocks)
        self.linear2_3 = nn.Linear(in_features=self.num_stocks**2,
                                   out_features=self.num_stocks)

        self.hidden_cell = (torch.zeros(self.num_layers,
                                        num_stocks,
                                        self.hidden_size).to(device),
                            torch.zeros(self.num_layers,
                                        num_stocks,
                                        self.hidden_size).to(device))

    def forward(self, x):

        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)

        y = lstm_out.flatten(start_dim=1)

        y = self.drop1(y)

        y = self.linear1_3(y)
        y = F.relu(y)

        y = self.drop2(y)

#        y = self.linear1_2(y)
#        y = F.relu(y)

#        y = self.linear1_3(y)
#        scores = F.relu(y)

        scores = y

        scores = scores.reshape(
            (x.shape[0], self.num_stocks, 1))

        p_hat = compute_permu_matrix(scores, tau=5)

        z = p_hat.flatten(start_dim=1)

#        z = self.linear2_1(z)
#        z = F.relu(z)

#        z = self.linear2_2(z)
#        z = F.relu(z)

#        z = self.linear2_3(z)
#        z = F.softmax(z, dim=1)

        z = self.drop3(z)

        z = self.linear2_3(z)
        z = F.softmax(z, dim=1)

        return z


train_losses = list()

plt.ion()
fig = plt.figure()
#ax = fig.gca()
plt.plot([12, 0, 3])
# plt.show()

# exit()


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0

    num_correct = 0

    for batch_idx, (x, y_true, actual) in enumerate(train_loader):

        print(f'len {len(x)}')

        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)
        actual = actual.to(device)

        open_true = actual[:, :, 0]
        close_true = actual[:, :, 3]
        rel_return_true = (close_true - open_true) / open_true

        rank_true = torch.argsort(rel_return_true)

        top_k_true = rank_true < top_k
        top_k_true = top_k_true.float()

        x = x.flatten(start_dim=2)

        model.hidden_cell = (torch.zeros(model.num_layers,
                                         train_batch_size,
                                         model.hidden_size).to(device),
                             torch.zeros(model.num_layers,
                                         train_batch_size,
                                         model.hidden_size).to(device))

        y = model(x)

        num_correct += prop_top_k_correct(y, rank_true < top_k, top_k)

        loss = F.binary_cross_entropy(y, top_k_true)

        loss.backward()
        optimizer.step()

        avg_loss += loss

    avg_loss *= train_batch_size / train_size

    train_losses.append(avg_loss)

    plt.clf()
    plt.plot(train_losses)
#    plt.show()
    plt.pause(0.01)

    print(f'Epoch {epoch}: avg. train loss: {avg_loss}')
#    print('\tTop-k correct: {} / {}'.format(
#        num_correct, num_stocks * train_size))


validation_losses = list()


def validate(model, device, validate_loader):

    model.eval()
    validate_loss = 0.

#    correct = 0
#    any_correct = 0.0

    with torch.no_grad():

        for x, y_true, actual in validate_loader:

            x, y_true = x.to(device), y_true.to(device)
            actual = actual.to(device)

            open_true = actual[:, :, 0]
            close_true = actual[:, :, 3]
            rel_return_true = (close_true - open_true) / open_true

            rank_true = torch.argsort(rel_return_true)

            top_k_true = rank_true < top_k
            top_k_true = top_k_true.float()

            x = x.flatten(start_dim=2)

            model.hidden_cell = (torch.zeros(model.num_layers,
                                             validate_batch_size,
                                             model.hidden_size).to(device),
                                 torch.zeros(model.num_layers,
                                             validate_batch_size,
                                             model.hidden_size).to(device))

            y = model(x)

            validate_loss += F.binary_cross_entropy(y, top_k_true)


#            y_true = y_true.reshape(validate_batch_size, num_stocks, 1)
#            y_true = y_true[:, 3::5].reshape(test_batch_size, num_stocks, 1)
#            x = torch.cat(tuple(x[:, :, i, :]
#                                for i in range(x.shape[2])),
#                          dim=0)

#            y = model(x)

#            y = torch.stack(tuple(y[(i * validate_batch_size):(i + 1) * validate_batch_size, :]
#                                  for i in range(num_stocks)), dim=1)

#            p_true = compute_permu_matrix(y_true, 1e-10)
#            p_hat = compute_permu_matrix(y, 5)

#            validate_loss += -torch.sum(p_true * torch.log(p_hat + 1e-20),
#                                        dim=1).mean()

#            correct += prop_correct(p_true, p_hat)
#            any_correct += prop_any_correct(p_true, p_hat)

    validate_loss *= validate_batch_size / validate_size

    validation_losses.append(validate_loss)

    plt.plot(validation_losses)
#    plt.show()
    plt.pause(0.01)

    print('\nValidation avg. loss: {:.4f}'.format(validate_loss))
#    print('  all correct: {:.0f} / {:.0f} = {:.1f}%'.format(
#        correct, len(validate_loader.dataset),
#        100. * correct / len(validate_loader.dataset)))
#    print('  any correct: {:.1f}%'.format(
#        100. * any_correct * validate_batch_size / len(validate_loader.dataset)))
#    print()


# Initialize train data loader.
print('initializing train loader...')
train_loader = DataLoader(
    WindowNorm(num_samples=train_size,
               mode='train',
               num_stocks=num_stocks,
               use_cache=True,
               forecast_window=forecast_window
               ),
    batch_size=train_batch_size,
    shuffle=True)
print('done')

# Initialize validation data loader.
print('initializing validation loader...')
validate_loader = DataLoader(
    WindowNorm(num_samples=validate_size,
               mode='validate',
               num_stocks=num_stocks,
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
               num_stocks=num_stocks,
               use_cache=True,
               forecast_window=forecast_window),
    batch_size=test_batch_size,
    shuffle=True)
print('done')

num_stocks = train_loader.dataset.num_stocks
num_attributes = train_loader.dataset.num_attributes

model = LstmModel3(num_layers=num_layers,
                   hidden_size=hidden_size,
                   num_stocks=num_stocks,
                   num_attributes=num_attributes,
                   device=device).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
    validate(model, device, validate_loader)
