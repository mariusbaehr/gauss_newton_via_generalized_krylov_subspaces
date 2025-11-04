import numpy as np
import scipy.sparse


parameter_count = 1000  # N=2p-2, again not a classical regression model, however


def res(x):
    block1 = 10 * (x[1:] - x[:-1] ** 2)
    block2 = 1 - x[:-1]
    return 2**0.5 * np.concatenate([block1, block2])


def jac(x):
    block1 = 10 * scipy.sparse.eye(
        parameter_count - 1, parameter_count, k=1
    ) - 20 * scipy.sparse.diags(x[:-1], shape=(parameter_count - 1, parameter_count))
    block2 = -scipy.sparse.eye(parameter_count - 1, parameter_count, k=0)
    return 2**0.5 * scipy.sparse.block_array([[block1], [block2]])


x_exact = np.ones(parameter_count)


def error(x):
    return np.linalg.norm(x - x_exact)
