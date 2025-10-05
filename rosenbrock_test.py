import numpy as np
import scipy.sparse
from benchmark import benchmark


p = 100  # N=2p-2, again not a classical regression model, however


def res(x):
    block1 = 10 * (x[1:] - x[:-1] ** 2)
    block2 = 1 - x[:-1]
    return np.concatenate([block1, block2])




def jac(x):
    block1 = 10 * scipy.sparse.eye(p - 1, p, k=1) - 20 * scipy.sparse.diags(
        x[:-1], shape=(p - 1, p)
    )
    block2 = -scipy.sparse.eye(p - 1, p, k=0)
    return scipy.sparse.block_array([[block1], [block2]])


x_exact = np.ones(p)


def error(x):
    return np.linalg.norm(x - x_exact)

from gauss_newton import gauss_newton

if __name__ == '__main__':

    x0 = np.zeros(p)
    x0[0] = 1.5  # Because zero is not allowed in gauss_newton_krylow
    benchmark(res, x0, jac, error, {"max_iter": 500}, title="rosenbrock_i_")
    
    x0_ii = gauss_newton(res, x0, jac,max_iter=155).x
    print(x0_ii)
    benchmark(res, x0_ii, jac, error, {"max_iter":500}, title="rosenbrock_ii_")

    x0_iii = 2*x_exact # (1+1E-1)*x_exact
    benchmark(res,x0_iii, jac, error, {"max_iter":500}, title="rosenbrock_iii_")

    x0_iv = x0_iii.copy()
    x0_iv[0] = 1
    
    benchmark(res,x0_iv, jac, error, {"max_iter":500}, title="rosenbrock_iv_")
