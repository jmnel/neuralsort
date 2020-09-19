from pprint import pprint

import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


def paa_compress(tx, N):
    b = tx.shape[0]
    n = tx.shape[-1]

    tx_bar = np.zeros((b, 2, N))

    for i in range(N):
        k = i + 1
        j0 = int((n / N) * (k - 1) + 1) - 1
        j1 = int((n / N) * k) - 1
        for q in range(b):
            tx_bar[q, 0, i] = (N / n) * np.sum(tx[q, 0, j0:j1 + 1])
            tx_bar[q, 1, i] = (N / n) * np.sum(tx[q, 1, j0:j1 + 1])

    return tx_bar
#    plt.plot(tx[0, 0], tx[0, 1])
#    plt.plot(tx_bar[0, 0], tx_bar[0, 1])
#    plt.show()
#    exit()
