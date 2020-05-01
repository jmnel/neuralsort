import numpy as np
from scipy import special
from scipy.stats import kurtosis
import scipy.optimize as optim
#import seaborn
#import matplotlib
# matplotlib.use('Qt5Cairo')
#import matplotlib.pyplot as plt
# plt.style.use('ggplot')

EPS = np.finfo(float).eps


def w_d(z, delta):
    """
    Calculates the bijection W_δ that is part of the inverse W_τ from (eq. 8).

    Args:
        z           center-scaled values from output distribution Y
        delta       heavy-tailness parameter

    """

    # Limit delta to machine precision.
    if delta < EPS:
        return z

    # Return bijection.
    return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z**2)) / delta)


def w_t(y, tau):
    """
    Invert distribution Y using parameters τ = (μₓ, σₓ, δ). See (eq. 9).

    Args:
        y       values from output heavy-tailed distribution Y
        tau     parameters of distribution transformation

    Returns:
        Inverted distribution X

    """

    return tau[0] + tau[1] * w_d((y - tau[0]) / tau[1], tau[2])


def delta_init(z):
    """
    Calculates initialization value for delta.


    Args:
        z       standardized heavy-tailed distribution

    """

    # Calculate empircal kurtosis of Z.
    gamma = kurtosis(z, fisher=False, bias=False)

    with np.errstate(all='ignore'):
        delta0 = np.clip(
            1. / 66. * (np.sqrt(66. * gamma - 162.) - 6.), 0.01, 0.48)

    if not np.isfinite(delta0):
        delta0 = 0.01

    return delta0


def delta_gmm(z):
    """
    Finds an optimal delta as a step for IGMM.

    Implements Algorithm 1 from (Georg et. al 2010)

    Args:
        z       standardized heavy-tailed distribution

    """

    # Initialize starting value for solver.
    delta0 = delta_init(z)

    # This enforces delta > 0.
    delta0 = np.log(delta0 + 0.001)

    # Define the objective function; this is the LP norm of kurtosis.
    def obj_fn(delta):

        # Enforce delta > 0.
        delta = np.exp(delta)

        # Back-transform Z to X.
        u = w_d(z, delta)

        if not np.all(np.isfinite(u)):
            return 0.

#        assert(np.all(np.isfinite(u)))

        # Calculate empirical kurtosis of back-transformed distribution.
        gamma_2 = kurtosis(u, fisher=False, bias=False)

#        assert np.isfinite(gamma_2)
        if not np.isfinite(gamma_2) or gamma_2 > 1e10:
            return 1e10

        # Return LP norm; 3 is kurtosis of normal distribution N(0, 1).
        return abs(gamma_2**2 - 3**2)

    # Optimize to find optimal delta.
    res = optim.minimize(obj_fn,
                         delta0,
                         method='Nelder-Mead')

    return np.exp(res.x[-1])


def igmm(y,
         tol=1e-6,
         max_iter=100):
    """
    Uses the Iterative Generalized Methods of Moments (IGMM) to estimate parameters τ.

    This implements Alogorithm 2 from (Goerg et. all).

    Args:
        y           Values from heavy-tailed output distribution Y
        tol         Stopping tolerance
        max_iter    Maximum number of iterations

    Returns:
        tau         Parameters which invert heavy-tailedness of Y to X

    """

    # Handle case of very small sigma.
    if np.std(y) < EPS:
        return np.mean(y), np.std(y).clip(EPS), 0.

    # Initialize delta.
    delta0 = delta_init(y)

    # Initialize tau with empircal values of distribution Y.
    tau1 = (np.median(y), np.std(y) * (1. - 2. * delta0) ** 0.75, delta0)

    for k in range(max_iter):
        tau0 = tau1

        # Standardize distribution Y to get Z.
        z = (y - tau1[0]) / tau1[1]

        # Find optimal delta via algorithm 1.
        delta1 = delta_gmm(z)

        assert z is not None
        assert delta1 is not None

        # Back-transform Y -> X.
        x = tau0[0] + tau1[1] * w_d(z, delta1)

        # Get empirical parameters of back-transformed data.
        mu1, sigma1, = np.mean(x), np.std(x)

        tau1 = (mu1, sigma1, delta1)

        # Check for stopping tolerance.
        if np.linalg.norm(np.array(tau1) - np.array(tau0)) < tol:
            break

        # Otherwise, check if max iterations have been reached.
        else:
            if k == max_iter - 1:
                pass
#                print(f'warning: no convergence after {max_iter} iterations.')

    return tau1


def gaussianize(y):
    """
    Gaussianizes a heavy-tailed distribution Y by estimating a inverted Lambert W X F transform.

    This simply wraps a call to the IGMM algorithm, and transforms by the estimated parameters τ.

    Args:
        y       Values of heavy-tailed distribution Y

    Returns:
        x       Inverted values of "Gaussianized" distribution X
        τ       Estimated parameters of Lambert W X F distribution


    """

    # Estimated parameters τ.
    tau = igmm(y)
    mu, sigma, delta = tau

    if sigma <= 0 or delta <= 0:
        raise ValueError('sigma or delta non-positive')

    # Return inverted distribution X and parameters τ used to perform inversion.
    return w_t(y, tau), tau


# n = 100000
# x = np.random.normal(0, 1, n)

# ax1 = plt.subplot(3, 2, 1)
# ax2 = plt.subplot(3, 2, 2)

# seaborn.distplot(x, bins=200, ax=ax1)

# y = x * np.exp((0.1 / 2) * x**2)

# y = np.random.laplace(0, 1, n)
# seaborn.distplot(y, bins=200, ax=ax2)

# k1 = kurtosis(x, fisher=False, bias=False)
# k2 = kurtosis(y, fisher=False, bias=False)

# x_hat, tau = gaussianize(y)

# ax3 = plt.subplot(3, 2, 3)
# seaborn.distplot(x_hat, bins=200, ax=ax3)


# delta_gmm(y)

# tau_est = igmm(y)

# print(tau_est)

# print(f'k1: {k1}, k2: {k2}')

# plt.show()
