import numpy as np
import scipy.sparse
import sympy as sp
from benchmark import benchmark

n = 101  # Notice N=P=(n-1)*(n-1)
alpha = 5
lamb = 10

a = -3
b = 3
h = (b - a) / n


def u(x1, x2):
    return np.exp(-10 * (x1**2 + x2**2))


A1 = scipy.sparse.diags_array(
    (-np.ones(n - 2), 2 * np.ones(n - 1), -np.ones(n - 2)), offsets=(-1, 0, 1)
)  # shape (n-1,n-1)
A2 = scipy.sparse.kron(A1, scipy.sparse.eye(n - 1)) + scipy.sparse.kron(
    scipy.sparse.eye(n - 1), A1
)  # shape ((n-1)**2,(n-1)**2)
Dx1 = scipy.sparse.kron(
    scipy.sparse.diags_array((-np.ones(n - 1), np.ones(n - 2)), offsets=(0, 1)),
    scipy.sparse.eye(n - 1),
)  # shape ((n-1)**2,(n-1)**2)

x1, x2 = np.meshgrid(np.linspace(a, b, n + 1)[1:-1], np.linspace(a, b, n + 1)[1:-1])

x_true = u(x1, x2).flatten("F")


def res(x, h, rhs, alpha, lamb):
    return rhs - (h**-2 * A2 @ x + h**-1 * alpha * Dx1 @ x + lamb * np.exp(x))


def jac(x, h, rhs, alpha, lamb):
    return -1 * (
        h**-2 * A2 + h**-1 * alpha * Dx1 + lamb * scipy.sparse.diags(np.exp(x))
    )


def error(x):
    return np.linalg.norm(x_true - x)


if __name__ == "__main__":

    x0 = x_true + 10**-1 * np.ones_like(x_true)

    y_true_h = (
        h**-2 * A2 @ x_true + h**-1 * alpha * Dx1 @ x_true + lamb * np.exp(x_true)
    )
    #    benchmark(
    #        res, x0, jac, error, {"args": (h,y_true_h,alpha,lamb), "max_iter": 100, "tol": 1e-12}, title="bratu_pde"
    #    )
    #
    y_true_1 = A2 @ x_true + alpha * Dx1 @ x_true + lamb * np.exp(x_true)

    #    benchmark(
    #        res, x0, jac, error, {"args": (1,y_true_1,alpha,lamb), "max_iter": 100, "tol": 1e-12}, title="bratu_pde_h"
    #    )
    #
    #    sp_x, sp_y = sp.symbols('sp_x sp_y')
    #    sp_u = sp.exp(-10*(sp_x**2+sp_y**2))
    #    sp_f = - sp.diff(sp_u,sp_x,sp_x) - sp.diff(sp_u,sp_y,sp_y) + alpha*sp.diff(sp_u,sp_x)  + lamb*sp.exp(sp_u)
    #    lamb_f = sp.lambdify((sp_x,sp_y),sp_f)
    #    y_manufactured_solution = lamb_f(x1, x2).flatten('F')
    #
    #    benchmark(
    #        res, x0, jac, error, {"args": (h,y_manufactured_solution,alpha ,lamb), "max_iter": 22, "tol": 1e-12}, title="bratu_manufactured_solution")
    #
    #    y_noise_h = y_true_h + np.random.default_rng(seed=42).normal(0,1,len(y_true_h))
    #    benchmark(
    #        res, x0, jac, error, {"args": (h,y_noise_h, alpha, lamb), "max_iter": 100, "tol": 1e-12}, title="bratu_noise_")

    y_true = A2 @ x_true + 0 * Dx1 @ x_true + 0 * np.exp(x_true)
    x0 = jac(x0, 1, y_true, 0, 0).T @ y_true  # Note that jac is constant
    x0 /= np.linalg.norm(x0)

    #    def aTa(x,h):
    #        return (h**-2*A2 + h**-1*alpha * Dx1).T @ ((h**-2*A2 + h**-1 * alpha * Dx1) @ x)
    #
    #    ATA1 = scipy.sparse.linalg.LinearOperator(((n - 1) ** 2, (n - 1) ** 2), matvec= lambda x:aTa(x,1))
    #    ATAh = scipy.sparse.linalg.LinearOperator(((n - 1) ** 2, (n - 1) ** 2), matvec= lambda x:aTa(x,h))
    #
    #    def ref_cg(callback):
    #        cg, _ = scipy.sparse.linalg.cg(
    #            ATA1, A2.T @ y_true + a1 * Dx1.T @ y_true, x0, callback=callback
    #        )
    #        return cg

    benchmark(
        res,
        x0,
        jac,
        error,
        {"args": (1, y_true, 0, 0), "max_iter": 100, "tol": 1e-12},
        title="bratu_linear_",
    )

# print(f"condition number of A.T@A {np.linalg.cond( ((A_h2+a1*D_x1).T@(A_h2+a1*D_x1)).todense())}")
# print(f"norm of A.T@A {np.linalg.norm(((A_h2+a1*D_x1).T@(A_h2+a1*D_x1)).todense(),2)}")
# print(f"condition number of A {np.linalg.cond((A_h2+a1*D_x1).todense())}")

# import matplotlib.pyplot as plt
# spec = np.linalg.eigvals(((A_h2+a1*d_x1).T@((A_h2+a1*d_x1))).todense())
# plt.figure()
# plt.scatter(np.real(spec),np.imag(spec))
# plt.scatter(0,0,label="Null")
# plt.show()
