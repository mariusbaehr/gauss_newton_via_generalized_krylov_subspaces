import numpy as np
import scipy.sparse
import sympy as sp
from benchmark import benchmark_method, ref_method
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow
import matplotlib.pyplot as plt
import statistics 

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "pgf.rcfonts": False,
    }
)

def default_u(x1,x2):
    return np.exp(-10 * (x1**2 + x2**2))

def make_problem(grid_nodes, grid_resolution=None, lower_bound=-3, upper_bound=3, u=default_u):
    """

    Notice that n=p=(grid_nodes-1)*(grid_nodes-1)

    """

    if grid_resolution == None:
        grid_resolution = (upper_bound - lower_bound) / grid_nodes

    A1 = scipy.sparse.diags_array(
        (-np.ones(grid_nodes - 2), 2 * np.ones(grid_nodes - 1), -np.ones(grid_nodes - 2)), offsets=(-1, 0, 1)
    )  # shape (grid_nodes-1,grid_nodes-1)

    A2 = scipy.sparse.kron(A1, scipy.sparse.eye(grid_nodes - 1)) + scipy.sparse.kron(
        scipy.sparse.eye(grid_nodes - 1), A1
    )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

    A2 *= grid_resolution**-2

    Dx1 = scipy.sparse.kron(
        scipy.sparse.diags_array((-np.ones(grid_nodes - 1), np.ones(grid_nodes - 2)), offsets=(0, 1)),
        scipy.sparse.eye(grid_nodes - 1),
    )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

    Dx1 *= grid_resolution**-1

    grid = np.meshgrid(np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1], np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1])
    u_true = u(*grid).flatten("F")

    return {
        "M": grid_nodes,
        "A": lower_bound,
        "B": upper_bound,
        "h": grid_resolution,
        "A2": A2,
        "Dx1": Dx1,
        "u_true": u_true,
        "grid": grid,
    }


def pde_operator(u, A2, Dx1, alpha, lamb):
    if lamb == 0:#To prevent overflow for the linear test case
        return A2 @ u + alpha * Dx1 @ u
    return A2 @ u + alpha * Dx1 @ u + lamb * np.exp(u)


def make_res(y, A2, Dx1, alpha, lamb):
    return lambda u: y - pde_operator(u, A2, Dx1, alpha, lamb)


def make_jac(A2, Dx1, alpha, lamb):
    if lamb == 0: #To prevent overflow for the linear test case
        return lambda u: -1 * (A2 + alpha * Dx1)

    return lambda u: -1 * (A2 + alpha * Dx1 + lamb * scipy.sparse.diags(np.exp(u)))


def make_error(u_true):
    return lambda u: np.linalg.norm(u_true - u)


def compare():
    grid_nodes = 101
    alpha = 5
    lamb = 10

    problem = make_problem(grid_nodes)
    A2, Dx1, u_true = problem["A2"], problem["Dx1"], problem["u_true"]

    y = pde_operator(u_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(u_true)
    np.random.seed(42)
    u0 = u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(u_true))

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

    print(f"Compare default, mean cg iter = {statistics.mean(gn_data[3])}")

    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_cg.pdf", bbox_inches="tight")
    plt.show()


def compare_without_scaling():
    grid_nodes = 101
    alpha = 5
    lamb = 10

    problem = make_problem(grid_nodes, grid_resolution=1)
    A2, Dx1, u_true = problem["A2"], problem["Dx1"], problem["u_true"]

    y = pde_operator(u_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(u_true)
    np.random.seed(42)
    u0 = u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(u_true))

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

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")
    
    plt.legend()
    plt.savefig("bratu_without_scaling_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_without_scaling_cg.pdf", bbox_inches="tight")
    plt.show()
    print(f"Compare without scaling, mean cg iter = {statistics.mean(gn_data[3])}")


def compare_manufactured_solution():
    grid_nodes = 101
    alpha = 5
    lamb = 10

    problem = make_problem(grid_nodes)
    A2, Dx1, u_true, grid = (
        problem["A2"],
        problem["Dx1"],
        problem["u_true"],
        problem["grid"],
    )

    sp_x, sp_y = sp.symbols("sp_x sp_y")
    sp_u = sp.exp(-10 * (sp_x**2 + sp_y**2))
    sp_f = (
        -sp.diff(sp_u, sp_x, sp_x)
        - sp.diff(sp_u, sp_y, sp_y)
        + alpha * sp.diff(sp_u, sp_x)
        + lamb * sp.exp(sp_u)
    )
    lamb_f = sp.lambdify((sp_x, sp_y), sp_f)
    y = lamb_f(*grid).flatten("F")

    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(u_true)
    np.random.seed(42)
    u0 = u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(u_true))

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
    alpha = 5
    lamb = 0

    problem = make_problem(grid_nodes, grid_resolution=None)
    A2, Dx1, u_true, grid_nodes = problem["A2"], problem["Dx1"], problem["u_true"], problem["M"]

    def cg_ref(res, u0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, u0, callback=cb_cg)

    y = pde_operator(u_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(u_true)

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


    def cb_for_breakdown(descent_direction):
        print(f" gnk algorithm, norm(descent_direction) = {np.linalg.norm(descent_direction)}")

    gauss_newton_krylow(res, u0, jac, callback=cb_for_breakdown)


    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.semilogy(cg_data[0][:100], "-", label="CG")
    print(f"Compare linear, count of cg iterations = {len(cg_data[0])}, final error = {cg_data[0][-1]}")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|u_k-u_h\|$")
    plt.legend()
    plt.savefig("bratu_linear_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.plot(cg_data[2], "-", label="CG")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")
    plt.legend()
    plt.savefig("bratu_linear_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_linear_cg.pdf", bbox_inches="tight")
    plt.show()

    print(f"Compare linear, mean cg iter = {statistics.mean(gn_data[3])}")


def compare_linear_small():
    grid_nodes = 25
    alpha = 5
    lamb = 0

    problem = make_problem(grid_nodes)
    A2, Dx1, u_true, grid_nodes = problem["A2"], problem["Dx1"], problem["u_true"], problem["M"]

    def cg_ref(res, u0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, u0, callback=cb_cg)

    y = pde_operator(u_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(u_true)

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

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(gn_data[2], "-s", label="Gauß-Newton")
    plt.plot(gnk_data[2], "-x", label="GNK")
    plt.plot(gnk_ii_data[2], "-+", label="GNK-(II)")
    plt.plot(ref_data[2], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Anzahl Residuums Auswertungen")
    plt.legend()
    plt.savefig("bratu_linear_small_nfev.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_linear_small_cg.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    compare()
    compare_without_scaling()
    #compare_manufactured_solution()
    compare_linear()
    compare_linear_small()
