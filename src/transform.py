import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
import scipy as sc

x = np.random.random(100000)

a1 = plt.subplot(2, 2, 1)
seaborn.distplot(x, ax=a1, bins=40)

#y = -np.sqrt(2) * sc.special.erfinv(2 * x)

y = np.sign(x - 0.5) * np.log(1 - 2 * np.abs(x - 0.5))

z = np.zeros_like(y)

for i in range(len(y)):
    if y[i] <= 0:
        z[i] = 0.5 * np.exp(y[i])
    else:
        z[i] = 1 - 0.5 * np.exp(-y[i])

#y = 0.5 * np.exp(y / 1)

a2 = plt.subplot(2, 2, 2)
seaborn.distplot(z, ax=a2, bins=40)

plt.show()
