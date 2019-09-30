import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from math import sin, cos, tan

NOISE_LEVEL = 0.3


def f(X, add_noise=False):
    comp_a = X[0] + 0.1 * sin(X[1])
    comp_b = - 0.8 * cos(X[2])
    comp_c = - tan(X[1]-X[0]) * 4
    result = comp_a + comp_b + comp_c
    if add_noise:
        result += np.random.uniform(0-NOISE_LEVEL, NOISE_LEVEL, 1)[0]
    return result


if __name__ == "__main__":

    X = np.random.rand(1200, 3)
    y = [f(v) for v in X]

    kernel = RBF([X.shape[0], X.shape[0], X.shape[0]], (1e-2, 1e2))
    # kernel = RBF([5, 5, 5], (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-1)
    gp.fit(X, y)

    # Generate test data that looks like a normal time serie
    x = []

    x1 = 0.5
    x2 = 0
    x3 = 0

    for i in range(0, 100):
        x1_prev, x2_prev, x3_prev = x1, x2, x3
        random_diffs = np.random.uniform(-0.1, 0.1, 3)
        x1 += random_diffs[0]
        x2 += random_diffs[1]
        x3 += random_diffs[2]

        if x1 < 0 or x1 > 1 or x2 < 0 or x2 > 1 or x3 < 0 or x3 > 1:
            x1, x2, x3 = x1_prev, x2_prev, x3_prev

        x += [[x1, x2, x3]]

    x = np.array(x)

    y_pred, MSE = gp.predict(x, return_std=True)

    y_expected = [f(v, add_noise=True) for v in x]
    idx = np.array(range(0, len(x)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.plot(idx, y_pred, c="blue", alpha=1.0, linewidth=0.5)
    ax.plot(idx, y_expected, c="red", alpha=0.5, linewidth=0.8)

    plt.show()
