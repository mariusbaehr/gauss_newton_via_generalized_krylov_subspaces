import numpy as np
import scipy.sparse
import sympy as sp
from benchmark import benchmark_method, ref_method
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "pgf.rcfonts": False,
    }
)


def make_problem(N, h=None, A=-3, B=3, u=lambda x1, x2: np.exp(-10 * (x1**2 + x2**2))):
    """

    Notice that n=p=(N-1)*(N-1)

    """

    if h == None:
        h = (B - A) / N

    A1 = scipy.sparse.diags_array(
        (-np.ones(N - 2), 2 * np.ones(N - 1), -np.ones(N - 2)), offsets=(-1, 0, 1)
    )  # shape (N-1,N-1)

    A2 = scipy.sparse.kron(A1, scipy.sparse.eye(N - 1)) + scipy.sparse.kron(
        scipy.sparse.eye(N - 1), A1
    )  # shape ((N-1)**2,(N-1)**2)

    A2 *= h**-2

    Dx1 = scipy.sparse.kron(
        scipy.sparse.diags_array((-np.ones(N - 1), np.ones(N - 2)), offsets=(0, 1)),
        scipy.sparse.eye(N - 1),
    )  # shape ((N-1)**2,(N-1)**2)

    Dx1 *= h**-1

    x1, x2 = np.meshgrid(np.linspace(A, B, N + 1)[1:-1], np.linspace(A, B, N + 1)[1:-1])
    grid = np.meshgrid(np.linspace(A, B, N + 1)[1:-1], np.linspace(A, B, N + 1)[1:-1])
    x_true = u(x1, x2).flatten("F")
    x_true = u(*grid).flatten("F")

    return {
        "N": N,
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
    N = 101
    alpha = 5
    lamb = 10

    problem = make_problem(N)
    A2, Dx1, x_true = problem["A2"], problem["Dx1"], problem["x_true"]

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)
    x0 = x_true + 10**-1 * np.ones_like(x_true)

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
    plt.semilogy(gn_data[0], "-x", label="gn")
    plt.semilogy(gnk_data[0], "-x", label="gnk")
    plt.semilogy(gnk_ii_data[0], "-x", label="gnk_ii")
    plt.semilogy(ref_data[0], "-x", label="ref")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()


def compare_without_scaling():
    N = 101
    alpha = 5
    lamb = 10

    problem = make_problem(N, h=1)
    A2, Dx1, x_true = problem["A2"], problem["Dx1"], problem["x_true"]

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)
    x0 = x_true + 10**-1 * np.ones_like(x_true)

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
    plt.semilogy(gn_data[0], "-x", label="gn")
    plt.semilogy(gnk_data[0], "-x", label="gnk")
    plt.semilogy(gnk_ii_data[0], "-x", label="gnk_ii")
    plt.semilogy(ref_data[0], "-x", label="ref")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()


def compare_manufactured_solution():
    N = 101
    alpha = 5
    lamb = 10

    problem = make_problem(N)
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
    x0 = x_true + 10**-1 * np.ones_like(x_true)

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
    plt.semilogy(gn_data[0], "-x", label="gn")
    plt.semilogy(gnk_data[0], "-x", label="gnk")
    plt.semilogy(gnk_ii_data[0], "-x", label="gnk_ii")
    plt.semilogy(ref_data[0], "-x", label="ref")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()


def compare_linear():
    N = 101
    alpha = 5
    lamb = 0

    problem = make_problem(N, h=None)
    A2, Dx1, x_true, N = problem["A2"], problem["Dx1"], problem["x_true"], problem["N"]

    def cg_ref(res, x0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((N - 1) ** 2, (N - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, x0, callback=cb_cg)

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)

    x0 = -1 * jac(np.zeros((N - 1) ** 2)).T @ y  # Note that jac is constant

    gn_data = benchmark_method(gauss_newton, res, x0, jac, error)
    gn_no_preconditioner_data = benchmark_method(
        gauss_newton, res, x0, jac, error, kwargs={"cg_preconditioner": False}
    )
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

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-x", label="gn")
    plt.semilogy(gnk_data[0], "-x", label="gnk")
    plt.semilogy(gnk_ii_data[0], "-+", label="gnk_ii")
    plt.semilogy(ref_data[0], "-x", label="ref")
    plt.semilogy(cg_data[0], "-x", label="cg")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(gn_data[2], "-x", label="gn")
    plt.plot(gnk_data[2], "-x", label="gnk")
    plt.plot(gnk_ii_data[2], "-+", label="gnk_ii")
    plt.plot(ref_data[2], "-x", label="ref")
    plt.plot(cg_data[2], "-x", label="cg")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Funktionsauswertungen")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-x", label="gn")
    plt.plot(gn_no_preconditioner_data[3], "-x", label="gn no prec")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.show()


def compare_linear_small():
    N = 25
    alpha = 5
    lamb = 0

    problem = make_problem(N)
    A2, Dx1, x_true, N = problem["A2"], problem["Dx1"], problem["x_true"], problem["N"]

    def cg_ref(res, x0, jac, args, callback):
        def cb_cg(x):
            callback(x, None, None)

        def aTa(x):
            return (A2 + alpha * Dx1).T @ ((A2 + alpha * Dx1) @ x)

        ATA = scipy.sparse.linalg.LinearOperator(((N - 1) ** 2, (N - 1) ** 2), aTa)
        scipy.sparse.linalg.cg(ATA, x0, callback=cb_cg)

    y = pde_operator(x_true, A2, Dx1, alpha, lamb)
    res = make_res(y, A2, Dx1, alpha, lamb)
    jac = make_jac(A2, Dx1, alpha, lamb)
    error = make_error(x_true)

    x0 = -1 * jac(np.zeros((N - 1) ** 2)).T @ y  # Note that jac is constant

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
    cg_data = benchmark_method(cg_ref, res, x0, jac, error)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogy(gn_data[0], "-x", label="gn")
    plt.semilogy(gnk_data[0], "-x", label="gnk")
    plt.semilogy(gnk_ii_data[0], "-+", label="gnk_ii")
    plt.semilogy(ref_data[0], "-x", label="ref")
    plt.semilogy(cg_data[0], "-x", label="cg")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9, 4), dpi=300)
    plt.plot(gn_data[2], "-x", label="gn")
    plt.plot(gnk_data[2], "-x", label="gnk")
    plt.plot(gnk_ii_data[2], "-+", label="gnk_ii")
    plt.plot(ref_data[2], "-x", label="ref")
    plt.plot(cg_data[2], "-x", label="cg")
    plt.xlabel("Iterationen")
    plt.ylabel(r"Funktionsauswertungen")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(gn_data[3], "-x", label="gn")
    plt.xlabel("Iterationen")
    plt.ylabel(r"CG Iterationen pro Iteration")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # compare()
    # compare_without_scaling()
    # compare_manufactured_solution()
    compare_linear()
    # compare_linear_small()
