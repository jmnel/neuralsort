import sqlite3
from pprint import pprint
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn

import settings

with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

    ts_msft = db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod_symbols_view
WHERE symbol = 'MSFT'
ORDER BY date;
''').fetchall()

    ts_uhn = db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod_symbols_view
WHERE symbol = 'GOOGL'
ORDER BY date;
''').fetchall()
dates, o, h, l, c, v = list(zip(*ts_uhn))

dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

window_len = 500

o = np.array(o)
o_ln = np.log(o / o[0])

n = len(o) - window_len

win_mu = np.zeros(n - window_len)
win_std = np.zeros(n - window_len)

for t in range(n - window_len):
    window = o_ln[t:t + n]

    rel_returns = np.diff(window)
    win_mu[t] = np.mean(rel_returns)
    win_std[t] = np.std(rel_returns)


fig, axs = plt.subplots(3, 1)
axs[0].plot(dates, o, lw=0.4)
axs[1].plot(o_ln, lw=0.4)
#axs[2].plot(win_mu, lw=0.4)
axs[2].plot(win_std, lw=0.4)
plt.show()
