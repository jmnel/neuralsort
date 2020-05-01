import sqlite3
from pprint import pprint
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn

import settings

with sqlite3.connect(settings.DATA_DIRECTORY / settings.DATABASE_NAME) as db:

    ts_msft = db.execute('''
SELECT date, adj_open, adj_high, adj_low, adj_close, adj_volume
FROM qdl_eod_symbols_view
WHERE symbol = 'GOOGL'
ORDER BY date;
''').fetchall()

dates, o, h, l, c, v = list(zip(*ts_msft))

dates = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

o = np.array(o)
#o = np.log(o / o[0])

t0 = 0
t1 = len(o)
v = np.array(v[t0:t1])
o = o[t0:t1]

o = np.log(o / o[0])

t_dom = np.arange(len(o))

alpha = 2
ema12 = np.zeros(len(o))
ema26 = np.zeros(len(o))

l1 = 12
l2 = 26

for t in range(len(o)):
    ema12[t] = o[t] * (alpha / (1 + l1))
    ema26[t] = o[t] * (alpha / (1 + l2))
    if t == 0:
        ema12[t] += o[0]
        ema26[t] += o[0]
    else:
        ema12[t] += ema12[t - 1] * (1 - alpha / (1 + l1))
        ema26[t] += ema26[t - 1] * (1 - alpha / (1 + l2))

mcad = ema12 - ema26


xlim = (500, 2000)

# plt.plot(o[2000:], lw=0.4)
fig, axs = plt.subplots(2, 1)
axs[0].plot(ema12, lw=0.4, label='EMA12')
axs[0].plot(ema26, lw=0.4, label='EMA26')
axs[0].plot(o, lw=0.4, label='GOOGL log-normed')
axs[0].set_xlim(xlim)
axs[0].set_ylim((0.8, 2.1))
axs[0].legend()

axs[1].plot(mcad, lw=1.0, label='MACD : EMA12 - EMA26')
# axs[1].set_xlim(xlim)
axs[1].set_xlim((xlim))
axs[1].set_ylim((-0.075, 0.075))
axs[1].fill_between(t_dom, 0, mcad)
axs[1].legend()
#axs[1].plot(v, lw=0.4)

plt.savefig('macd.png', dpi=200)
plt.show()


# ema26 = 0

# o = np.log(o / o[0])


# o_diff = np.diff(o)

# a_cor = np.correlate(o, o, mode='full')

# print(len(o_diff))
# print(len(a_cor))

# plt.plot(a_cor, lw=0.4)

# plt.scatter(dates[1:], o_diff, s=0.4)

# seaborn.distplot(o_diff, bins=200)
# plt.hist(o_diff, bins=200, histtype='step')

# print(np.mean(o_diff))
# plt.plot(dates, o, lw=0.5)
# plt.show()

# print(o[0])
