import sqlite3
from random import shuffle
import json

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import seaborn

import settings
from quantgan.quantgan_dataset import QuantGanDataset

with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

    rows = db.execute(
        'SELECT symbol, meta_json FROM quantgan_meta;').fetchall()

    shuffle(rows)

    rows = rows[:6]

    print(len(rows))

    symbols = [row[0] for row in rows]
    meta = [json.loads(row[1]) for row in rows]

    data = dict()

    for sym in symbols:

        rows = db.execute('''
SELECT log_return, x_norm1, x_gaus, x_norm2 FROM quantgan_data
WHERE symbol == ? ORDER BY date;
''', (sym,)).fetchall()

        log_return, x_norm1, x_gaus, x_norm2 = list(zip(*rows))

        data[sym] = (log_return, x_norm1, x_gaus, x_norm2)

    log_ret_comb = list()
    x_norm1_comb = list()

#    ax = tuple(plt.subplot(3, 2, i) for i in range(1, 7))
    for idx, (sym, x) in enumerate(data.items()):

        #        p = seaborn.distplot(x[0], bins=200, ax=ax[idx],
        #                             label=f'log-return: {sym}',
        #                             color=f'C{idx}')
        #        ax[idx].set_title(f'{sym}')

        log_ret_comb += x[0]
        x_norm1_comb += x[1]

    sym = symbols[0]
    seaborn.distplot(data[sym][0], bins=100)

#    plt.tight_layout()
    plt.savefig('../figures/figure1-1.png')

    plt.clf()

    log_return = data[sym][0]
    x_norm2 = data[sym][3]

    seaborn.distplot(x_norm2, bins=100, color='C1')
    plt.savefig('../figures/figure1-2.png')

    print(meta[0])
