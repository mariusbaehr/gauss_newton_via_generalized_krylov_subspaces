from collections.abc import Callable
from typing import Tuple, Any
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np

class StepLengthConvergenceError(RuntimeError):
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def armijo_goldstein(
    res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
    x: npt.NDArray,
    res_ev: npt.NDArray,
    jac_ev: npt.NDArray | sp.spmatrix,
    args: Tuple,
    descent_direction: npt.NDArray,
    max_iter: int = 100,
    initial_step_length: float = 1.0,
) -> Tuple[float, npt.NDArray, int]:
    """
    Performs the Armijo Goldstein rule for determining a suitable step length.

    Parameters
    ----------
    res: Residual funciton.
    x: Current argument value.
    res_ev: Evalueated residual at x.
    jac_ev: Evaluated jacobian of the residual at x.
    descent_direction: Descent direction.
    max_iter: Maximum iteration count.
    initial_step_length: Initial step size.

    Returns
    -------
    step_lenth: The determined step length.
    current_res: The residual evaluated at the point x + step_length * descent_direction.
    iters: Number of iteration used.

    """

    step_length: float = initial_step_length

    prev_loss: float = np.sum(res_ev**2)
    jac_dot_descent: float = np.sum((jac_ev @ descent_direction) ** 2)
    success: bool = False
    iter: int = 0

    for iter in range(max_iter):

        current_res = res(x + step_length * descent_direction, *args)
        current_loss: float = np.sum(current_res**2)
        if prev_loss - current_loss >= 0.5 * step_length * jac_dot_descent:
            success = True
            break
        else:
            step_length /= 2

    if not success:
        raise StepLengthConvergenceError(
            "The armijio_goldstein subroutine reached maximum iteration bound before principle was satisfied! Possible reasons:"
            + "\n- The max iteration count is not big enough to allow for a sufficiently small step size"
            + "\n- Or the descent direction is invalid."
            + f"Norm of descent_direction ={np.linalg.norm(descent_direction)}."
        )

    return step_length, current_res, iter + 1
