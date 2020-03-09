import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent / 'libs'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

from db_connectors import SQLite3Connector
from alphavantage import RelativeReturnsDataset
from lstm_model import LstmModel

torch.manual_seed(0)
device = torch.device('cuda')

num_layers = 4

train_size = 6400
test_size = 1600
train_batch_size = 200
test_batch_size = 200

epochs = 1000
forecast_window = 10
num_seqences = 5

top_k = 2


def _bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1).to(device)
    b = _bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0).to(device)
    c = _bl_matmul(s, d)
    mat_p = (c - b).permute(0, 2, 1)
    mat_p = F.softmax(mat_p / tau, -1)

    return mat_p


def _prop_any_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2).float()
    correct = torch.mean(eq, axis=-1)
    return torch.mean(correct)


def _prop_correct(p1, p2):
    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2)
    correct = torch.all(eq, axis=-1).float()
    return torch.sum(correct)


def train(model, device, train_loader, optimizer, epoch):

    model.train()

    avg_loss = 0

    for batch_idx, (seq, label) in enumerate(train_loader):
        optimizer.zero_grad()

        seq = seq.to(device) * 1
        label = label.to(device) * 1

        model.hidden_cell = (torch.zeros(num_layers, train_batch_size, 100).to(device),
                             torch.zeros(num_layers, train_batch_size, 100).to(device))

#        print(seq.is_cuda)
#        print(label.is_cuda)

        scores = model(seq)
        scores = scores.reshape(train_batch_size, num_seqences, 1)

        true_scores = label

        p_true = compute_permu_matrix(true_scores, 1e-10)
        p_hat = compute_permu_matrix(scores, 5)

#        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20),
#                          dim=1).mean()

        foo = torch.argsort(true_scores, dim=1, descending=True)
        foo = foo.reshape((train_batch_size, num_seqences))

#        p_bar = torch.zeros(p_true.shape).to(device)

#        print(foo.shape##)
#        print(foo)
#        u, v = torch.meshgrid(foo[:, :top_k], foo[:, :top_k])

#        u = torch.zeros((train_batch_size, 2, 2)).long()
#        v = torch.zeros((train_batch_size, 2, 2)).long()

#        for kq in range(train_batch_size):
#            u[kq], v[kq] = torch.meshgrid(foo[kq, :top_k], foo[kq, :top_k])
#            u[1], v[1] = torch.meshgrid(foo[1, :top_k], foo[1, :top_k])
#        print('u=')
#        print(u)

#        p_bar[:, u, v] = 1.0 / top_k

#        s = torch.zeros((train_batch_size, 3, 3)).long()
#        t = torch.zeros((train_batch_size, 3, 3)).long()

#        for kq in range(train_batch_size):
#            s[kq], t[kq] = torch.meshgrid(foo[kq, top_k:], foo[kq, top_k:])
#        s[1], t[1] = torch.meshgrid(foo[1, top_k:], foo[1, top_k:])

#        p_bar[:, s, t] = 1.0 / (num_seqences - top_k)

        loss = -torch.sum(p_true * torch.log(p_hat + 1e-20),
                          dim=1).mean()

#        print(loss)

        avg_loss += loss

        loss.backward()
        optimizer.step()

    avg_loss = avg_loss * train_batch_size / train_size

    print(f'epoch {epoch} avg loss: {avg_loss}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0

    correct = 0
    any_correct = 0.0

    with torch.no_grad():

        for seq, label in test_loader:

            seq, label = seq.to(device) * 1, label.to(device) * 1

            model.hidden_cell = (torch.zeros(num_layers, test_batch_size, 100).to(device),
                                 torch.zeros(num_layers, test_batch_size, 100).to(device))

            scores = model(seq)

            scores = scores.reshape(test_batch_size, num_seqences, 1)
            true_scores = label

            p_true = compute_permu_matrix(true_scores, 1e-10)
            p_hat = compute_permu_matrix(scores, 5)

            correct += _prop_correct(p_true, p_hat)
            any_correct += _prop_any_correct(p_true, p_hat)

            test_loss += -torch.sum(p_true *
                                    torch.log(p_hat + 1e-20), dim=1).mean()

    test_loss *= test_batch_size / test_size

    print('\nTest avg. loss: {:.4f}'.format(test_loss))
    print('  all correct: {:.0f} / {:.0f} = {:.1f}%'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('  any correct: {:.1f}%'.format(
        100. * any_correct * test_batch_size / len(test_loader.dataset)))
    print()


data_path = Path(__file__).absolute().parents[2] / 'data'

train_loader = torch.utils.data.DataLoader(
    RelativeReturnsDataset(data_path,
                           train_size,
                           forecast_window,
                           num_seqences,
                           train=True),
    batch_size=train_batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    RelativeReturnsDataset(data_path,
                           test_size,
                           forecast_window,
                           num_seqences,
                           train=False),
    batch_size=test_batch_size, shuffle=True)

model = LstmModel(num_layers=num_layers, num_seqences=num_seqences)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):

    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
