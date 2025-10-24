import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from benchmark import benchmark_method, ref_method
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "pgf.rcfonts": False,
    }
)

p = 100  # N=2p-2, again not a classical regression model, however


def res(x):
    block1 = 10 * (x[1:] - x[:-1] ** 2)
    block2 = 1 - x[:-1]
    return 2**0.5*np.concatenate([block1, block2])


def jac(x):
    block1 = 10 * scipy.sparse.eye(p - 1, p, k=1) - 20 * scipy.sparse.diags(
        x[:-1], shape=(p - 1, p)
    )
    block2 = -scipy.sparse.eye(p - 1, p, k=0)
    return 2**0.5*scipy.sparse.block_array([[block1], [block2]])


x_exact = np.ones(p)


def error(x):
    return np.linalg.norm(x - x_exact)


def compare_error_i():
    np.random.seed(42)
    x0 = x_exact + 0.1 * np.random.normal(loc=0, scale=1, size=p)

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gnk_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error)
    gnk_ii_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error, kwargs = {"version":"res_new"})
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    fig1, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
    ax1.set_xlabel("Iterationen")
    ax1.set_ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
    ax2.set_xlabel("Iterationen")
    ax2.set_ylabel(r"Verlust $\log\mathcal{L}$")
    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=300)
    ax3.set_xlabel("Iterationen")
    ax3.set_ylabel(r"Anzahl $f$ Auswertungen")
    fig4, ax4 = plt.subplots(figsize=(8, 4), dpi=300)
    ax4.set_xlabel("Iterationen")
    ax4.set_ylabel(r"Anzahl CG-Iterationen")


    ax1.semilogy(gn_data[0],"-x",label="gn")
    ax1.semilogy(gnk_data[0],"-x",label="gnk")
    ax1.semilogy(gnk_ii_data[0],"-x",label="gnk_ii")
    ax1.semilogy(ref_data[0],"-x",label="ref")


    ax2.semilogy(gn_data[1],"-x",label="gn")
    ax2.semilogy(gnk_data[1],"-x",label="gnk")
    ax2.semilogy(gnk_ii_data[1],"-x",label="gnk_ii")
    ax2.semilogy(ref_data[1],"-x",label="ref")


    ax3.plot(gn_data[2],"-x",label="gn")
    ax3.plot(gnk_data[2],"-x",label="gnk")
    ax3.plot(gnk_ii_data[2],"-x",label="gnk_ii")
    ax3.plot(ref_data[2],"-x",label="ref")

    ax4.plot(gn_data[3],"-x",label="gn")


    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    plt.show()


def compare_error_ii():
    x0 = 2 * x_exact

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gnk_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error)
    gnk_ii_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error, kwargs = {"version":"res_new"})
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    plt.figure(figsize=(8,4), dpi =300)
    plt.semilogy(gn_data[0],"-x",label="gn")
    plt.semilogy(gnk_data[0],"-x",label="gnk")
    plt.semilogy(gnk_ii_data[0],"-x",label="gnk_ii")
    plt.semilogy(ref_data[0],"-x",label="ref")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()

def compare_error_iii():

    x0 = 2 * x_exact
    x0[0] = 1

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gnk_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error)
    gnk_ii_data = benchmark_method(gauss_newton_krylow, res, x0, jac, error, kwargs = {"version":"res_new"})
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    plt.figure(figsize=(8,4), dpi =300)
    plt.semilogy(gn_data[0],"-x",label="gn")
    plt.semilogy(gnk_data[0],"-x",label="gnk")
    plt.semilogy(gnk_ii_data[0],"-x",label="gnk_ii")
    plt.semilogy(ref_data[0],"-x",label="ref")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    compare_error_i()
    compare_error_ii()
    compare_error_iii()

