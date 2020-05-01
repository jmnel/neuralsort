from pprint import pprint
from random import shuffle, randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import matplotlib
matplotlib.use('Qt4agg')
import matplotlib.pyplot as plt

from datasets.gan_examples import GanExamples

torch.manual_seed(0)

batch_size = 100

# Get CUDA if it is available and set as device.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

ngf = 2


class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        nz = 1
        nc = 1

        self.conv1 = nn.ConvTranspose1d(nz, ngf * 128, 4, 1, 0, bias=True)
        self.norm1 = nn.BatchNorm1d(ngf * 128)

        self.conv2 = nn.ConvTranspose1d(
            ngf * 128, ngf * 64, 4, 2, 1, bias=True)
        self.norm2 = nn.BatchNorm1d(ngf * 64)

        self.conv3 = nn.ConvTranspose1d(
            ngf * 64, ngf * 32, 4, 2, 1, bias=True)
        self.norm3 = nn.BatchNorm1d(ngf * 32)

        self.conv4 = nn.ConvTranspose1d(
            ngf * 32, ngf * 16, 4, 2, 1, bias=True)
        self.norm4 = nn.BatchNorm1d(ngf * 16)

        self.conv5 = nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=True)
        self.norm5 = nn.BatchNorm1d(ngf * 8)

        self.conv6 = nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=True)
        self.norm6 = nn.BatchNorm1d(ngf * 4)

        self.conv7 = nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=True)
        self.norm7 = nn.BatchNorm1d(ngf * 2)

        self.conv8 = nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=True)
        self.norm8 = nn.BatchNorm1d(ngf)

        self.conv9 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=True)
        self.norm9 = nn.BatchNorm1d(ngf)

        self.conv10 = nn.ConvTranspose1d(ngf, ngf, 4, 2, 1, bias=True)
        self.norm10 = nn.BatchNorm1d(ngf)

        self.conv11 = nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=True)

        self.fc1 = nn.Linear(4096, 4096 * 4)
        self.fc2 = nn.Linear(4096 * 4, 4096 * 2)
        self.fc3 = nn.Linear(4096 * 2, 4096)

    def forward(self, z):

        x = self.conv1(z)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.norm7(x)
        x = F.relu(x)

        x = self.conv8(x)
        x = self.norm8(x)
        x = F.relu(x)

        x = self.conv9(x)
        x = self.norm9(x)
        x = F.relu(x)

        x = self.conv10(x)
        x = self.norm10(x)
        x = F.relu(x)

        x = self.conv11(x)

#        x = torch.flatten(x, start_dim=1)
#        x = self.fc1(x)
#        x = F.relu(x)
#        x = self.fc2(x)
#        x = F.relu(x)
#        x = self.fc3(x)

        x = x.reshape((batch_size, 4096))
        return x
#        x = torch.tanh(x)

#        w = torch.FloatTensor(x.shape)
#        for i in range(4096):
#            w[:, 0, i] = torch.prod(x[:, 0, :i], axis=-1)

#        return x[:, 0, :]


class Descriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, ngf * 4, 4, 1)
        self.conv2 = nn.Conv1d(ngf * 4, ngf * 8, 4, 1, 0)

        self.conv3 = nn.Conv1d(ngf * 8, ngf * 16, 4, 1, 0)
        self.conv4 = nn.Conv1d(ngf * 16, ngf * 32, 4, 1, 0)

        self.conv5 = nn.Conv1d(ngf * 32, ngf * 64, 4, 1, 0)
        self.conv6 = nn.Conv1d(ngf * 64, ngf * 128, 4, 1, 0)

        self.fc1 = nn.Linear(65216, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool1d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = F.max_pool1d(x, 2)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

#        return x
#        print(x)

        return torch.sigmoid(x)

#        x = F.max_pool1d(x, 2)

#        print(x.shape)
#        exit()


plt.ion()
fig = plt.figure()
ax = fig.gca()
plt.show()


class Gan:

    def __init__(self, epochs):

        self.epochs = epochs

        self.dataloader = DataLoader(
            dataset=GanExamples(),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True)

        self.generator = Generator()
        self.descriminator = Descriminator()

        self.genr_optim = optim.Adam(
            lr=1e-4, params=self.generator.parameters())
        self.desc_optim = optim.Adam(
            lr=1e-4, params=self.descriminator.parameters())

    def train(self,
              generator,
              descriminator,
              data_loader,
              generator_optimizer,
              descrimnator_optimizer,
              epoch):

        for idx, x_example in enumerate(data_loader):

            if idx % 10 == 0:
                print(
                    f'EPOCH {epoch} : {idx * data_loader.batch_size} / {len(data_loader)}')

            generator.train()
            descriminator.train()

            generator_optimizer.zero_grad()
            descrimnator_optimizer.zero_grad()

            z = torch.randn((batch_size, 1, 1))
            x_gen = generator(z)

#            print(x_gen.shape)

            desc_loss = 0.0

            perm = randint(0, 1)

            x_example = x_example.float()

            xs = (x_example, x_gen)

            s0 = descriminator(x_example.view((batch_size, 1, 4096))).flatten()
            s1 = descriminator(x_gen.view((batch_size, 1, 4096))).flatten()

#            if s0 > 0 and s1 < 0:
#                print('desc is right')
#            else:
#                print('desc wrong')

            g_loss = torch.sum(-s1)

            d_loss = torch.sum(-s0 + s1)

            g_loss.backward(retain_graph=True)

            descrimnator_optimizer.zero_grad()

            d_loss.backward()

            print(f'g-loss: {g_loss.item()}')
            print(f'd-loss: {d_loss.item()}\n')

            generator_optimizer.step()
            descrimnator_optimizer.step()

            plt.clf()
            plt.plot(x_example[0].numpy(), lw=0.4)
            plt.plot(x_gen[0].detach().numpy(), lw=0.4)
            plt.ylim((-3, 6))
            plt.pause(0.01)

    def run(self):
        for epoch in range(self.epochs):
            self.train(self.generator,
                       self.descriminator,
                       self.dataloader,
                       self.genr_optim,
                       self.desc_optim,
                       epoch)


EPOCHS = 2000
gan = Gan(EPOCHS)
gan.run()
