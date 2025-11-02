import numpy as np
import scipy.sparse
import sympy as sp
from benchmark import benchmark_method, ref_method
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow
from typing import Optional, Callable
import numpy.typing as npt
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

class BratuPdeProblem:
    """ 
    Creates Bratu PDE problem instance, necessary because different starting values, parameters and grid_nodes are used.

    Notice that n=p=(grid_nodes-1)*(grid_nodes-1).
    """
    grid_nodes: int

    def __init__(
            self,
            grid_nodes: int,
            ALPHA: float,
            LAMBDA: float,
            lower_bound: float = -3.0,
            upper_bound: float = 3.0,
            grid_resolution: Optional[float] = None,
            u: Callable[[npt.NDArray,npt.NDArray],npt.NDArray] = default_u):
        self.grid_nodes = grid_nodes
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.grid_resolution: float
        if grid_resolution == None:
            self.grid_resolution = (upper_bound - lower_bound) / grid_nodes
        else:
            self.grid_resolution = grid_resolution
        self.u = u
        
        self.laplace1d = scipy.sparse.diags_array(
            (-np.ones(grid_nodes - 2), 2 * np.ones(grid_nodes - 1), -np.ones(grid_nodes - 2)), offsets=(-1, 0, 1)
        )  # shape (grid_nodes-1,grid_nodes-1)

        self.laplace2d = scipy.sparse.kron(self.laplace1d, scipy.sparse.eye(grid_nodes - 1)) + scipy.sparse.kron(
            scipy.sparse.eye(grid_nodes - 1), self.laplace1d 
        )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

        self.laplace2d *= self.grid_resolution**-2

        self.partial_diff_x = scipy.sparse.kron(
            scipy.sparse.diags_array((-np.ones(grid_nodes - 1), np.ones(grid_nodes - 2)), offsets=(0, 1)),
            scipy.sparse.eye(grid_nodes - 1),
        )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

        self.partial_diff_x *= self.grid_resolution**-1

        self.grid = np.meshgrid(np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1], np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1])

        self.u_true = u(*self.grid).flatten("F")

    def pde_operator(self,u:npt.NDArray):
        if self.LAMBDA == 0:#To prevent overflow for the linear test case
            return self.laplace2d @ u + self.ALPHA * self.partial_diff_x @ u
        return self.laplace2d @ u + self.ALPHA * self.partial_diff_x @ u + self.LAMBDA * np.exp(u)

    def make_res(self,y:npt.NDArray):
        return lambda u: y - self.pde_operator(u)

    def make_jac(self):
        if self.LAMBDA == 0: #To prevent overflow for the linear test case
            return lambda u: -1 * (self.laplace2d + self.ALPHA * self.partial_diff_x)

        return lambda u: -1 * (self.laplace2d + self.ALPHA * self.partial_diff_x + self.LAMBDA * scipy.sparse.diags(np.exp(u)))

    def make_error(self):
        return lambda u: np.linalg.norm(self.u_true - u)


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
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(bratu_pde.u_true))

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
    ALPHA = 5
    LAMBDA = 10

    bratu_pde = BratuPdeProblem(grid_nodes, ALPHA, LAMBDA,grid_resolution = 1)
    y = bratu_pde.pde_operator(bratu_pde.u_true)
    res = bratu_pde.make_res(y)
    jac = bratu_pde.make_jac()
    error = bratu_pde.make_error()

    np.random.seed(42)
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(bratu_pde.u_true))

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
    u0 = bratu_pde.u_true + 0.1 * np.random.normal(loc=0, scale=1, size=len(bratu_pde.u_true))

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
            return (bratu_pde.laplace2d + bratu_pde.ALPHA * bratu_pde.partial_diff_x).T @ ((bratu_pde.laplace2d + ALPHA * bratu_pde.partial_diff_x) @ u)

        ATA = scipy.sparse.linalg.LinearOperator(((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa)
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
            return (bratu_pde.laplace2d + bratu_pde.ALPHA * bratu_pde.partial_diff_x).T @ ((bratu_pde.laplace2d + ALPHA * bratu_pde.partial_diff_x) @ u)

        ATA = scipy.sparse.linalg.LinearOperator(((grid_nodes - 1) ** 2, (grid_nodes - 1) ** 2), aTa)
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
