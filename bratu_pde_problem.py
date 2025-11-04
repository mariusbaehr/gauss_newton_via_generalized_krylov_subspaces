import numpy as np
import scipy.sparse
from typing import Optional, Callable
import numpy.typing as npt

def default_u(x1, x2):
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
        u: Callable[[npt.NDArray, npt.NDArray], npt.NDArray] = default_u,
    ):
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
            (
                -np.ones(grid_nodes - 2),
                2 * np.ones(grid_nodes - 1),
                -np.ones(grid_nodes - 2),
            ),
            offsets=(-1, 0, 1),
        )  # shape (grid_nodes-1,grid_nodes-1)

        self.laplace2d = scipy.sparse.kron(
            self.laplace1d, scipy.sparse.eye(grid_nodes - 1)
        ) + scipy.sparse.kron(
            scipy.sparse.eye(grid_nodes - 1), self.laplace1d
        )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

        self.laplace2d *= self.grid_resolution**-2

        self.partial_diff_x = scipy.sparse.kron(
            scipy.sparse.diags_array(
                (-np.ones(grid_nodes - 1), np.ones(grid_nodes - 2)), offsets=(0, 1)
            ),
            scipy.sparse.eye(grid_nodes - 1),
        )  # shape ((grid_nodes-1)**2,(grid_nodes-1)**2)

        self.partial_diff_x *= self.grid_resolution**-1

        self.grid = np.meshgrid(
            np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1],
            np.linspace(lower_bound, upper_bound, grid_nodes + 1)[1:-1],
        )

        self.u_true = u(*self.grid).flatten("F")

    def pde_operator(self, u: npt.NDArray):
        if self.LAMBDA == 0:  # To prevent overflow for the linear test case
            return self.laplace2d @ u + self.ALPHA * self.partial_diff_x @ u
        return (
            self.laplace2d @ u
            + self.ALPHA * self.partial_diff_x @ u
            + self.LAMBDA * np.exp(u)
        )

    def make_res(self, y: npt.NDArray):
        return lambda u: y - self.pde_operator(u)

    def make_jac(self):
        if self.LAMBDA == 0:  # To prevent overflow for the linear test case
            return lambda u: -1 * (self.laplace2d + self.ALPHA * self.partial_diff_x)

        return lambda u: -1 * (
            self.laplace2d
            + self.ALPHA * self.partial_diff_x
            + self.LAMBDA * scipy.sparse.diags(np.exp(u))
        )

    def make_error(self):
        return lambda u: np.linalg.norm(self.u_true - u)
