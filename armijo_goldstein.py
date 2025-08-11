from collections.abc import Callable
import numpy.typing as npt
import scipy.sparse as sp
import numpy as np

def armijo_goldstein(res: Callable[[npt.NDArray],npt.NDArray], beta: npt.NDArray, jac_ev: npt.NDArray | sp.spmatrix, descent_direction: npt.NDArray, max_iter: int=1000, step_length0: float = 1.0) -> float:
    """
    Performs the Armijo Goldstein rule for determining a suitable step length.

    Parameters
    ----------
    res:
    beta:
    jac_ev:
    descent_direction: 
    max_iter:
    step_length0:

    Returns
    -------
    step_lenth: 


    """

    step_length: float = step_length0

    prev_loss: float = np.sum(res(beta)**2)
    jac_dot_descent: float = np.sum((jac_ev@descent_direction)**2)/2

    for _ in range(max_iter):

        current_loss: float = np.sum(res(beta+step_length*descent_direction)**2)
        if prev_loss - current_loss >= step_length*jac_dot_descent:
            return step_length
        else: 
            step_length /= 2

    return step_length

