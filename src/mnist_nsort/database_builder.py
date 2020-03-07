#!/usr/bin/env python3

from random import randint
import torch
from torchvision import datasets, transforms
import pickle
#import json
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTK3Cairo')

NUM_IMAGES = 10000
STITCH_SIZE = 4
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', 'data')


# class NumpyEncoder(json.JSONEncoder):
#    def default(self, obj):
#        if isinstance(obj, np.ndarray):
#            return obj.tolist()
#        return json.JSONEncoder.default(self, obj)


class MnistDatasetBuilder:

    def __init__(self):
        self.build_dataset()

    def load_glyphs(self):

        BATCH_SIZE = 60000

        print(
            f'  Loading {BATCH_SIZE} glyphs and labels from MNIST dataset...')

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(DATA_PATH, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])
                           ),
            batch_size=BATCH_SIZE, shuffle=True)

        batch_data, batch_labels = next(iter(train_loader))

        all_glphs = list()
        glyphs_nonzero = list()

        print('  Done')

        print('  Enumerating glyphs...')

        for i_glyph in range(BATCH_SIZE):

            if i_glyph % 20000 == 0:
                print('  {:.0f}% ...'.format(
                    100. * i_glyph / BATCH_SIZE))

            glyph = batch_data[i_glyph]
            label = batch_labels[i_glyph]

            if label != 0:
                glyphs_nonzero.append((glyph, label))

            all_glphs.append((glyph, label))

        print('  100% ... Done')

        return (all_glphs, glyphs_nonzero)

    def build_dataset(self):

        glyphs_all, glyphs_nonzero = self.load_glyphs()
        dataset = self.stitch_glyphs((glyphs_all, glyphs_nonzero))
        self.save_to_file(dataset)

    def stitch_glyphs(self, glyphs):

        glyphs_all, glyphs_nonzero = glyphs

        dataset = list()

        print(f'  Creating {NUM_IMAGES} {STITCH_SIZE}-stitched images...')

        for i in range(NUM_IMAGES):

            if i % (NUM_IMAGES // 4) == 0:
                print('  {:.0f}% ...'.format(
                    i * 100. / NUM_IMAGES
                ))

            stitch = torch.FloatTensor()
            stitch_label = 0

            for j in range(STITCH_SIZE):
                if j == STITCH_SIZE - 1:
                    glyph, label = glyphs_all[randint(0, len(glyphs_all) - 1)]
                else:
                    glyph, label = glyphs_nonzero[randint(
                        0, len(glyphs_nonzero) - 1)]

                stitch = torch.cat([
                    stitch,
                    glyph[0]
                ], 0)

                stitch_label += label * 10**(STITCH_SIZE - j - 1)

            dataset.append((stitch.cpu().numpy(), stitch_label.cpu().numpy()))

        print('  100% ... Done')

        return dataset

    def save_to_file(self, dataset):

        print('Saving dataset to file...')
#        with open(os.path.join(DATA_PATH, 'mnist_stiched.json'), 'w') as datafile:
#            json.dump(dataset, datafile, cls=NumpyEncoder)

        pickle.dump(dataset, open(os.path.join(
            DATA_PATH, 'mnist_stitched.pkl'), 'wb'))
        print('Done')


db = MnistDatasetBuilder()
