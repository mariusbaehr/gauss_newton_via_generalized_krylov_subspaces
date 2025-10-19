import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from gauss_newton import gauss_newton


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "pgf.rcfonts": False,
    }
)

p = 2


def res(x):
    block1 = 10 * (x[1:] - x[:-1] ** 2)
    block2 = 1 - x[:-1]
    return 2**0.5*np.concatenate([block1, block2])


x0 = np.array([-1.0, 1.0])


def jac(x):
    block1 = 10 * scipy.sparse.eye(p - 1, p, k=1) - 20 * scipy.sparse.diags(
        x[:-1], shape=(p - 1, p)
    )
    block2 = -scipy.sparse.eye(p - 1, p, k=0)
    return 2**0.5*scipy.sparse.block_array(
        [[block1], [block2]]
    ).todense()  # todense() to ensure the classical gauss-newton method is used, with direct least squares solver.


def loss(x):
    return 1/2*np.sum(res(x) ** 2)


@np.vectorize
def loss_vectorized(x1, x2):
    x = np.array([x1, x2])
    return loss(x)


x1, x2 = np.meshgrid(np.linspace(-1.25, 1.25, 250), np.linspace(-1.25, 1.25, 250))

loss_ev = loss_vectorized(x1, x2)


levels = np.linspace(loss_ev.min(), loss_ev.max(), 30)

plt.figure(figsize=(8, 4), dpi=300)
contourplot = plt.contour(x1, x2, loss_ev, levels=levels, cmap=cm.coolwarm)
cbar = plt.colorbar(contourplot, label=r"$\mathcal{L}$")
plt.plot(1, 1, "sk")
plt.text(1, 1.05, r"$x^\ast$")
plt.plot(*x0, "sk")
plt.text(*x0 + np.array([0, 0.05]), r"$x_0$")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")

x_list = [x0]


def cb_x(x):
    global error_list, loss_list
    x_list.append(x.copy())


gauss_newton(res, x0, jac, callback=cb_x)

x_array = np.array(x_list).T

plt.plot(x_array[0], x_array[1], "o-k", label="gauss newton")

plt.savefig("rosenbrock_3d.png", bbox_inches="tight")
plt.show()
