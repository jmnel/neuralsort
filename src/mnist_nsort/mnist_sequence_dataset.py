from random import randint

import os.path as path
import torch
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from time import perf_counter

import numpy as np

import matplotlib.pyplot as plt


class MnistSequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 num_stitched,
                 seq_length,
                 size,
                 train=True,
                 #                 train_size,
                 #                 test_size,
                 transform=None):
        super().__init__()

        self.transform = transform

        assert(num_stitched > 0)
        assert(seq_length > 1)

        data_path = path.dirname(path.realpath(__file__))
        data_path = path.join(data_path, '..', 'data')

        file_name = 'mnist_seq_train.pkl' if train else 'mnist_seq_test.pkl'
        save_path = path.join(data_path, file_name)

        file_exists = path.exists(save_path)

#        print(f'file exists={file_exists}')
#        print(f'path={data_path}')

        mnist_data = datasets.MNIST(data_path, train=train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]))

        mnist_batch_size = len(mnist_data)

        t_start = perf_counter()
        mnist_dataloader = DataLoader(mnist_data,
                                      batch_size=mnist_batch_size,
                                      shuffle=True,
                                      num_workers=0
                                      )

        mnist_img, mnist_labels = next(iter(mnist_dataloader))
        print(perf_counter() - t_start)
        mnist_img = mnist_img.reshape((mnist_batch_size, 28, 28))

        def gen_stitch():
            picks = tuple(randint(0, mnist_batch_size - 1)
                          for k in range(num_stitched))

            stitch_img = torch.cat([
                mnist_img[p] for p in picks], 1)

            stitch_label = sum(
                mnist_labels[picks[k]] * 10**(num_stitched - k - 1)
                for k in range(num_stitched))

            return (stitch_img, stitch_label)

        def gen_sequence():
            seq = tuple(gen_stitch() for j in range(seq_length))
            seq_img = torch.stack([s[0] for s in seq], 0)

#            print(f'img shape={seq_img.shape}')

            seq_labels = torch.tensor([s[1] for s in seq])

#            perm = torch.argsort(seq_labels)

            return (seq_img, seq_labels)

#        a, b = gen_sequence()
#        exit()

        print(
            f'  Generating {size} {seq_length}-sequences of {num_stitched}-stitches...')
        self.data = list([gen_sequence() for i in range(size)])
        print('Done')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx][0]
        label = self.data[idx][1]

        if self.transform:
            img = self.transform(img)

        return (img, label)


foo = MnistSequenceDataset(num_stitched=4, seq_length=5,
                           size=100)
