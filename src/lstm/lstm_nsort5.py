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

from datasets.test_dataset import TestDataset
from nsort import compute_permu_matrix, prop_correct, prop_any_correct

torch.manual_seed(0)

# Get CUDA if it is available and set as device.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

num_stocks = 50
num_attributes = 5
# top_k = 3
num_layers = 1
hidden_size = 100
# train_size = 6000
# validate_size = 2000
# test_size = 2000

train_batch_size = 200
validate_batch_size = 100
test_batch_size = 100

epochs = 2000
forecast_window = 100
hold_len = 10


def prop_top_k_correct(r, r_true, k):
    r = torch.argsort(r)
    r = r < top_k
    return torch.sum(r == r_true).item()


class LstmModel5(nn.Module):

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_stocks,
                 num_attributes,
                 forecast_len,
                 device):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_stocks = num_stocks
        self.num_attributes = num_attributes
        self.forecast_len = forecast_len

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear1_1 = nn.Linear(in_features=forecast_len * num_attributes,
                                   out_features=1000)

        self.linear1_2 = nn.Linear(in_features=1000,
                                   out_features=1000)

        self.linear1_3 = nn.Linear(in_features=1000,
                                   out_features=100)

        self.linear2_1 = nn.Linear(in_features=hidden_size,
                                   out_features=100)

        self.linear2_2 = nn.Linear(in_features=100,
                                   out_features=50)

        self.linear2_3 = nn.Linear(in_features=50,
                                   out_features=1)

    def forward(self, x):

        batch_size = x.shape[0]

        x = torch.cat([x[:, :, i, :] for i in range(self.num_stocks)], axis=0)

        x = x.flatten(start_dim=1)

        x = self.linear1_1(x)
        x = F.relu(x)

        x = self.linear1_2(x)
        x = F.relu(x)

        x = self.linear1_3(x)
        x = F.relu(x)
        x = x.view((batch_size * num_stocks, forecast_window, 1))

        lstm_out, self.hidden_cell = self.lstm(x)

        y = lstm_out[:, -1, :]

        y = self.linear2_1(y)
        y = F.relu(y)

        y = self.linear2_2(y)
        y = F.relu(y)

        y = self.linear2_3(y)
#        y = F.log_softmax(y)

        y = y.view(batch_size, num_stocks)

        return y

#        p_hat = compute_permu_matrix(y, tau=5)

#        return p_hat


class CnnModel(nn.Module):

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_stocks,
                 num_attributes,
                 forecast_len,
                 device):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_stocks = num_stocks
        self.num_attributes = num_attributes
        self.forecast_len = forecast_len

        self.conv1 = nn.Conv1d(5, 5 * 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(5 * 32, 5 * 64, kernel_size=3, stride=1)

        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(15360, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):

        batch_size = x.shape[0]

#        print(x.shape)

        x = torch.cat([x[:, :, i, :] for i in range(self.num_stocks)])

#        print(x.shape)

        x = torch.transpose(x, 1, 2)

#        print(x.shape)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc2(x).view((batch_size, num_stocks))

        return x
#        p_hat = compute_permu_matrix(x, tau=5)
#        return p_hat

#        print(x.shape)

#        exit()


plt.ion()
fig = plt.figure()
ax = fig.gca()
# plt.plot([12, 0, 3])
plt.show()

# exit()

train_losses = list()


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0

#    num_correct = 0

    for batch_idx, (x, y_true) in enumerate(train_loader):

        #        for i in range(num_stocks):
        #            plt.plot(x[0, :, i, 0], lw=0.4)
        #        plt.show()

        optimizer.zero_grad()
        x, y_true = x.to(device), y_true.to(device)

#        print(y_true.shape)

        model.hidden_cell = (torch.zeros(model.num_layers,
                                         train_batch_size,
                                         model.hidden_size).to(device),
                             torch.zeros(model.num_layers,
                                         train_batch_size,
                                         model.hidden_size).to(device))

#        p_hat = model(x)

        y = model(x)

        y_true = y_true.view(train_batch_size, num_stocks)

#        exit()

#        print(y_true)

        loss = F.mse_loss(y, y_true)

#        p_true = compute_permu_matrix(y_true, tau=1e-10)

#        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20),
#                          dim=1).mean()

        loss.backward()
        optimizer.step()
#        exit()

        avg_loss += loss

    avg_loss *= train_batch_size / len(train_loader)

    train_losses.append(avg_loss)

    plt.clf()
    plt.plot(train_losses)
#    plt.show()
    plt.pause(0.01)

    print(f'Epoch {epoch}: avg. train loss: {avg_loss}')
#    print('\tTop-k correct: {} / {}'.format(
#        num_correct, num_stocks * train_size))


# Initialize train data loader.
print('initializing data loader...')
dataset = TestDataset(num_stocks,
                      split_ratio=(8, 1, 1),
                      forecast_len=100,
                      hold_len=hold_len)


train_loader = DataLoader(
    dataset=dataset.train_view(),
    batch_size=train_batch_size,
    shuffle=True,
    drop_last=True)
validate_loader = DataLoader(
    dataset=dataset.validate_view(),
    batch_size=validate_batch_size,
    shuffle=True)
test_loader = DataLoader(
    dataset=dataset.test_view(),
    batch_size=test_batch_size,
    shuffle=True)

model = LstmModel5(num_layers=num_layers,
                   # model = CnnModel(num_layers=num_layers,
                   hidden_size=hidden_size,
                   num_stocks=num_stocks,
                   num_attributes=num_attributes,
                   forecast_len=forecast_window,
                   device=device).to(device)


optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, epoch)
#    validate(model, device, validate_loader)
