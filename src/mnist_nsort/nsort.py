from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from deepcnn import DeepCnn
from mnist_sequence_dataset import MnistSequenceDataset
# from mnist_stitched_dataset import StitchedMNIST

torch.manual_seed(0)
device = torch.device('cpu')

TRAIN_SIZE = 6400
TEST_SIZE = 1600
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 200
NUM_DIGITS = 4
NUM_SEQUENCES = 3


def _bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1)
    b = _bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0)
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

    avg_loss = 0.

    t_start = perf_counter()

    for batch_idx, (x, y) in enumerate(train_loader):

        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        scores = torch.zeros(BATCH_SIZE, NUM_SEQUENCES)

        for i in range(BATCH_SIZE):

            foo = model(x[i:i + 1, :, :].reshape((3, 1, 28, 112))
                        ).reshape((1, NUM_SEQUENCES))
            scores[i] = foo

        scores = scores.reshape((BATCH_SIZE, NUM_SEQUENCES, 1))

        true_scores = y.reshape((BATCH_SIZE, NUM_SEQUENCES, 1))
        true_scores = true_scores.type(torch.FloatTensor)

        p_true = compute_permu_matrix(true_scores, 1e-10)
        p_hat = compute_permu_matrix(scores, 5)

        loss = -torch.sum(p_true * torch.log(p_hat +
                                             1e-20), dim=2).mean()

        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
            ))

    t_elapsed = perf_counter() - t_start
    print('Train took {:.1f} seconds'.format(t_elapsed))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0

    correct = 0
    any_correct = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            scores = torch.zeros(TEST_BATCH_SIZE, NUM_SEQUENCES)

            for i in range(TEST_BATCH_SIZE):

                for j in range(NUM_SEQUENCES):
                    s = model(x[i:i + 1, j:j + 1, :, :])

                    scores[i, j] = s

            scores = scores.reshape((TEST_BATCH_SIZE, NUM_SEQUENCES, 1))
            true_scores = y.reshape((TEST_BATCH_SIZE, NUM_SEQUENCES, 1))
            true_scores = true_scores.type(torch.FloatTensor)

            p_true = compute_permu_matrix(true_scores, 1e-10)
            p_hat = compute_permu_matrix(scores, 5)

            correct += _prop_correct(p_true, p_hat)
            any_correct += _prop_any_correct(p_true, p_hat)

            test_loss = - \
                torch.sum(p_true * torch.log(p_hat + 1e-20), dim=1).mean()

    print('\nTest avg. loss: {:.4f}'.format(test_loss))
    print('  all correct: {:.0f} / {:.0f} = {:.1f}%'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('  any correct: {:.1f}%'.format(
        100. * any_correct * TEST_BATCH_SIZE / len(test_loader.dataset)))


train_loader = torch.utils.data.DataLoader(
    MnistSequenceDataset(
        num_stitched=4, seq_length=3,
        size=TRAIN_SIZE),
    batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    MnistSequenceDataset(
        num_stitched=4, seq_length=3,
        train=False, size=TEST_SIZE),
    batch_size=TEST_BATCH_SIZE, shuffle=False)

model = DeepCnn(num_digits=NUM_DIGITS)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
