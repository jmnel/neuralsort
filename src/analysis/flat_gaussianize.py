import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn
from scipy.stats import kurtosis

from lambertw.gaussianize import gaussianize


def normalize(y):
    return (y - np.mean(y)) / np.std(y)


n = 500000
y_laplace_1 = np.random.laplace(0, 4, size=n)
y_student_1 = np.random.standard_t(2, size=n)
y_cauchy_1 = np.random.standard_cauchy(size=n)

x_laplace_1 = normalize(gaussianize(normalize(y_laplace_1))[0])
x_student_1 = normalize(gaussianize(normalize(y_student_1))[0])
x_cauchy_1 = normalize(gaussianize(normalize(y_cauchy_1))[0])

axs = tuple(plt.subplot(3, 2, i) for i in range(1, 7))

seaborn.distplot(y_laplace_1, bins=200, ax=axs[0])
axs[0].set_title('Fig. 1.1: Laplace : σ = 4')
seaborn.distplot(x_laplace_1, bins=200, ax=axs[1], color='C1')
axs[1].set_title('Fig. 1.2: "Gaussianized" Laplace : σ = 4')

seaborn.distplot(y_student_1, bins=200, ax=axs[2], color='C2')
axs[2].set_title('Fig. 2.1: Student t : ν = 2')

seaborn.distplot(x_student_1, bins=200, ax=axs[3], color='C3')
axs[3].set_title('Fig. 2.2: "Gaussianized" Student t : ν = 300')

seaborn.distplot(y_cauchy_1, bins=200, ax=axs[4], color='C4')
axs[4].set_title('Fig. 3.1: Standard Cauchy')

seaborn.distplot(x_cauchy_1, bins=200, ax=axs[5], color='C5')
axs[5].set_title('Fig. 3.2: "Gaussianized" Standard Cauchy')

plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.savefig('flat_gaussianize.png', dpi=180)
# plt.show()
