import numpy as np
import scipy.sparse
from benchmark import benchmark


p = 100  # N=2p-2, again not a classical regression model, however


def res(x):
    block1 = 10 * (x[1:] - x[:-1] ** 2)
    block2 = 1 - x[:-1]
    return np.concatenate([block1, block2])


x0 = np.zeros(p)
x0[0] = 1  # Because zero is not allowed in gauss_newton_krylov


def jac(x):
    block1 = 10 * scipy.sparse.eye(p - 1, p, k=1) - 20 * scipy.sparse.diags(
        x[:-1], shape=(p - 1, p)
    )
    block2 = -scipy.sparse.eye(p - 1, p, k=0)
    return scipy.sparse.block_array([[block1], [block2]])


x_exact = np.ones(p)


def error(x):
    return np.linalg.norm(x - x_exact)


benchmark(res, x0, jac, error, {"max_iter": 500}, title="rosenbrock")

# TODO: plot the step length, I guess they are set to 1 when the error goes down
