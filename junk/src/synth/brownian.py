import numpy as np
import matplotlib
matplotlib.use('Qt5Cairo')
import matplotlib.pyplot as plt

mu = 0.01
n = 400
dt = 1e-2
x0 = 100
# np.random.seed(1)

# sigma = np.arange(0.01, 2, 0.2)
sigma = np.linspace(0.05, 0.4, 3)

x = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt),
                               size=(len(sigma), n)).T
)

x = np.vstack([np.ones(len(sigma)), x])
x = x0 * x.cumprod(axis=0)


ax = plt.gca()
ax.plot(x, linewidth=0.8)
ax.legend()
plt.show()
