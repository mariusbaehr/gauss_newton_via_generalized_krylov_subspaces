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


def make_problem(M, h=None, A=-3, B=3, u=lambda x1, x2: np.exp(-10 * (x1**2 + x2**2))):
    """

    Notice that n=p=(M-1)*(M-1)

    """

    if h == None:
        h = (B - A) / M

    A1 = scipy.sparse.diags_array(
        (-np.ones(M - 2), 2 * np.ones(M - 1), -np.ones(M - 2)), offsets=(-1, 0, 1)
    )  # shape (M-1,M-1)

    A2 = scipy.sparse.kron(A1, scipy.sparse.eye(M - 1)) + scipy.sparse.kron(
        scipy.sparse.eye(M - 1), A1
    )  # shape ((M-1)**2,(M-1)**2)

    A2 *= h**-2

    Dx1 = scipy.sparse.kron(
        scipy.sparse.diags_array((-np.ones(M - 1), np.ones(M - 2)), offsets=(0, 1)),
        scipy.sparse.eye(M - 1),
    )  # shape ((M-1)**2,(M-1)**2)

    Dx1 *= h**-1

    x1, x2 = np.meshgrid(np.linspace(A, B, M + 1)[1:-1], np.linspace(A, B, M + 1)[1:-1])
    grid = np.meshgrid(np.linspace(A, B, M + 1)[1:-1], np.linspace(A, B, M + 1)[1:-1])
    x_true = u(x1, x2).flatten("F")
    x_true = u(*grid).flatten("F")

    return {
        "M": M,
        "A": A,
        "B": B,
        "h": h,
        "A2": A2,
        "Dx1": Dx1,
        "x_true": x_true,
        "grid": grid,
    }


def pde_operator(x, A2, Dx1, alpha, lamb):
    if lamb == 0:
        return A2 @ x + alpha * Dx1 @ x
    return A2 @ x + alpha * Dx1 @ x + lamb * np.exp(x)


def make_res(y, A2, Dx1, alpha, lamb):
    return lambda x: y - pde_operator(x, A2, Dx1, alpha, lamb)


def make_jac(A2, Dx1, alpha, lamb):
    if lamb == 0:
        return lambda x: -1 * (A2 + alpha * Dx1)

    return lambda x: -1 * (A2 + alpha * Dx1 + lamb * scipy.sparse.diags(np.exp(x)))


def make_error(x_true):
    return lambda x: np.linalg.norm(x_true - x)


def compare():
    M = 101
    alpha = 5
    lamb = 10

    problem = make_problem(M)
    A2, Dx1, x_true = problem["A2"], problem["Dx1"], problem["x_true"]

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)
    x0 = x_true + 10**-1 * np.ones_like(x_true)
    np.random.seed(42)
    x0 = x_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(x_true))

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    #gn_no_preconditioner_data = benchmark_method( gauss_newton, res, x0, jac, error, kwargs={"cg_preconditioner": False})
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, x0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        x0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.savefig("bratu_error.pdf", bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-s", label="Gauß-Newton")
    #plt.plot(gn_no_preconditioner_data[3], "-x", label="gn no prec")

    print(f"Compare default, mean cg iter = {statistics.mean(gn_data[3])}")

    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_cg.pdf", bbox_inches="tight")
    plt.show()


def compare_without_scaling():
    M = 101
    alpha = 5
    lamb = 10

    problem = make_problem(M, h=1)
    A2, Dx1, x_true = problem["A2"], problem["Dx1"], problem["x_true"]

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)
    np.random.seed(42)
    x0 = x_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(x_true))

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    #gn_no_preconditioner_data = benchmark_method( gauss_newton, res, x0, jac, error, kwargs={"cg_preconditioner": False})
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, x0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        x0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
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
    #plt.plot(gn_no_preconditioner_data[3], "-x", label="gn no prec")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_without_scaling_cg.pdf", bbox_inches="tight")
    plt.show()
    print(f"Compare without scaling, mean cg iter = {statistics.mean(gn_data[3])}")


def compare_manufactured_solution():
    M = 101
    alpha = 5
    lamb = 10

    problem = make_problem(M)
    A2, Dx1, x_true, grid = (
        problem["A2"],
        problem["Dx1"],
        problem["x_true"],
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
    error = make_error(x_true)
    np.random.seed(42)
    x0 = x_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(x_true))

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, x0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        x0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, x0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()


def compare_linear():
    M = 101
    alpha = 5
    lamb = 0

    problem = make_problem(M, h=None)
    A2, Dx1, x_true, M = problem["A2"], problem["Dx1"], problem["x_true"], problem["M"]

    def cg_ref(res, x0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((M - 1) ** 2, (M - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, x0, callback=cb_cg)

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)

    x0 = -1 * jac(np.zeros((M - 1) ** 2)).T @ y  # Note that jac is constant

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    #gn_no_preconditioner_data = benchmark_method( gauss_newton, res, x0, jac, error, kwargs={"cg_preconditioner": False})
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, x0, jac, error, kwargs={"max_iter": 100}
    )
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        x0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 100},
    )
    ref_data = benchmark_method(ref_method, res, x0, jac, error)
    cg_data = benchmark_method(cg_ref, res, x0, jac, error)


    def cb_for_breakdown(descent_direction):
        print(f" gnk algorithm, norm(descent_direction) = {np.linalg.norm(descent_direction)}")

    gauss_newton_krylow(res, x0, jac, callback=cb_for_breakdown)


    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.semilogy(cg_data[0][:100], "-", label="CG")
    print(f"Compare linear, count of cg iterations = {len(cg_data[0])}, final error = {cg_data[0][-1]}")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
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
    #plt.plot(gn_no_preconditioner_data[3], "-x", label="gn no prec")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.savefig("bratu_linear_cg.pdf", bbox_inches="tight")
    plt.show()

    print(f"Compare linear, mean cg iter = {statistics.mean(gn_data[3])}")
    #print(f"Compare linear, mean cg iter = {statistics.mean(gn_no_preconditioner_data[3])}")


def compare_linear_small():
    M = 25
    alpha = 5
    lamb = 0

    problem = make_problem(M)
    A2, Dx1, x_true, M = problem["A2"], problem["Dx1"], problem["x_true"], problem["M"]

    def cg_ref(res, x0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((M - 1) ** 2, (M - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, x0, callback=cb_cg)

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)

    x0 = -1 * jac(np.zeros((M - 1) ** 2)).T @ y  # Note that jac is constant

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gnk_data = benchmark_method(
        gauss_newton_krylow, res, x0, jac, error, kwargs={"max_iter": 100}
    )
    #gn_no_preconditioner_data = benchmark_method( gauss_newton, res, x0, jac, error, kwargs={"cg_preconditioner": False})
    gnk_ii_data = benchmark_method(
        gauss_newton_krylow,
        res,
        x0,
        jac,
        error,
        kwargs={"version": "res_new", "max_iter": 200},
    )
    ref_data = benchmark_method(ref_method, res, x0, jac, error)
    cg_data = benchmark_method(cg_ref, res, x0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-s", label="Gauß-Newton")
    plt.semilogy(gnk_data[0], "-x", label="GNK")
    plt.semilogy(gnk_ii_data[0], "-+", label="GNK-(II)")
    plt.semilogy(ref_data[0], ".-", label="Referenz")
    plt.semilogy(cg_data[0], "-", label="CG")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
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
    #plt.plot(gn_no_preconditioner_data[3], "-x", label="Gauß-Newton")
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
