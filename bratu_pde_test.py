import numpy as np
import scipy.sparse
import sympy as sp
from benchmark import benchmark_method, ref_method
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow
from bratu_pde_problem import BratuPdeProblem
import matplotlib.pyplot as plt
import statistics
import matplotlib.ticker as ticker

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "pgf.rcfonts": False,
    }
)


def compare():
    grid_nodes = 101
    ALPHA = 5
    LAMBDA = 10

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA)

    y = bratu_pde.pde_operator(bratu_pde.u_true)
    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    np.random.seed(42)
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(
        loc=0, scale=1, size=len(bratu_pde.u_true)
    )

    gn_data = benchmark_method(gauss_newton, res, u0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, u0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        u0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, u0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.savefig("bratu_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    print(f"Compare default, mean cg iter = {statistics.mean(gn_data[3])}")

    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_cg.pdf", bbox_inches="tight")
    plt.show()


def compare_without_scaling():
    grid_nodes = 101
    ALPHA = 5
    LAMBDA = 10

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA, grid_resolution=1)
    y = bratu_pde.pde_operator(bratu_pde.u_true)
    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    np.random.seed(42)
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(
        loc=0, scale=1, size=len(bratu_pde.u_true)
    )

    gn_data = benchmark_method(gauss_newton, res, u0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, u0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        u0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, u0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.savefig("bratu_without_scaling_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 3), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")

    plt.legend()
    plt.savefig("bratu_without_scaling_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.savefig("bratu_without_scaling_cg.pdf", bbox_inches="tight")
    plt.show()
    print(f"Compare without scaling, mean cg iter = {statistics.mean(gn_data[3])}")


def compare_manufactured_solution():
    grid_nodes = 101
    ALPHA = 5
    LAMBDA = 10

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA)

    sp_x, sp_y = sp.symbols("sp_x sp_y")
    sp_u = sp.exp(-10 * (sp_x**2 + sp_y**2))
    sp_f = (
        -sp.diff(sp_u, sp_x, sp_x)
        - sp.diff(sp_u, sp_y, sp_y)
        + ALPHA * sp.diff(sp_u, sp_x)
        + LAMBDA * sp.exp(sp_u)
    )
    lamb_f = sp.lambdify((sp_x, sp_y), sp_f)
    y = lamb_f(*bratu_pde.grid).flatten("F")

    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    np.random.seed(42)
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(
        loc=0, scale=1, size=len(bratu_pde.u_true)
    )

    gn_data = benchmark_method(gauss_newton, res, u0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, u0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        u0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, u0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.show()


def compare_linear():
    grid_nodes = 101
    ALPHA = 5
    LAMBDA = 0

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA)

    y = bratu_pde.pde_operator(bratu_pde.u_true)
    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    def cg_ref(res, u0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(u):
            return (
                bratu_pde.laplace2d + bratu_pde.ALPHA * bratu_pde.partial_diff_x
            ).T @ ((bratu_pde.laplace2d + ALPHA * bratu_pde.partial_diff_x) @ u)

        ATA = scipy.sparse.linalg.LinearOperator(
            ((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa
        )
        scipy.sparse.linalg.cg(ATA, u0, callback=cb_cg)

    u0 = -1 * jac(np.zeros((grid_nodes - 1) ** 2)).T @ y  # Note that jac is constant

    gn_data = benchmark_method(gauss_newton, res, u0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, u0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        u0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, u0, jac, error)
    cg_data = benchmark_method(cg_ref, res, u0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.semilogy(cg_data[0][:100], "-", label="CG")
    print(
        f"Compare linear, count of cg iterations = {len(cg_data[0])}, final error = {cg_data[0][-1]}"
    )
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.savefig("bratu_linear_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 3), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.plot(cg_data[2], "-", label="CG")
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")
    plt.legend()
    plt.savefig("bratu_linear_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.savefig("bratu_linear_cg.pdf", bbox_inches="tight")
    plt.show()

    print(f"Compare linear, mean cg iter = {statistics.mean(gn_data[3])}")


def compare_linear_small():
    grid_nodes = 25
    ALPHA = 5
    LAMBDA = 0

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA)

    y = bratu_pde.pde_operator(bratu_pde.u_true)
    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    def cg_ref(res, u0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(u):
            return (
                bratu_pde.laplace2d + bratu_pde.ALPHA * bratu_pde.partial_diff_x
            ).T @ ((bratu_pde.laplace2d + ALPHA * bratu_pde.partial_diff_x) @ u)

        ATA = scipy.sparse.linalg.LinearOperator(
            ((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa
        )
        scipy.sparse.linalg.cg(ATA, u0, callback=cb_cg)

    u0 = -1 * jac(np.zeros((grid_nodes - 1) ** 2)).T @ y  # Note that jac is constant

    gn_data = benchmark_method(gauss_newton, res, u0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, u0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        u0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 200},
    )
    ref_data = benchmark_method(ref_method, res, u0, jac, error)
    cg_data = benchmark_method(cg_ref, res, u0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.semilogy(cg_data[0], "-", label="CG")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.savefig("bratu_linear_small_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 3), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")
    plt.legend()
    plt.savefig("bratu_linear_small_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.savefig("bratu_linear_small_cg.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    compare()
    compare_without_scaling()
    # compare_manufactured_solution()
    compare_linear()
    compare_linear_small()
