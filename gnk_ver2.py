from collections.abc import Callable
from typing import Union, Any, Tuple 
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy

def gauss_newton_krylov(
        res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray], 
        x0: npt.NDArray, 
        jac: Callable[[npt.NDArray,Tuple[Any]],Union[npt.NDArray,sp.spmatrix,sp.sparray]],
        krylov_restart: int | None = None,
        args: Tuple = (),
        tol: float=1E-8,
        max_iter=100,
        callback: Callable=lambda : None)->RegressionResult:
    """
    Parameters
    ----------
    res: The residual function, called as res(x, *args) the argument x and its return must always be ndarrays.
    x0: Initial guess of the regression parameters.
    jac: Called as jac(x,d, *args), should return either a NDArray or a spmatrix, if returned a spmatrix a sparse solver is used to solve the linearised equation.
    args: Additional arguments passed to res and jac.
    tol: Tolerance for termination by the change of the paramters x.
    max_iter: Maximum number of iterations.
    callback: Called as callback with the following possible positional arguments: x, iter, jac, rank_jac, step_length, nfev. Implementet arguments are automaticaly determined by call_callback.

    Returns
    -------
    res: Returns instance of RegressionResult.
    """

    if np.allclose(x0,np.zeros_like(x0)):
        raise ValueError("x0 is not allowed to be 0 in the gauss_newton_krylov algorithm")

    success: bool = False
    rank_jac: int | None = None
    krylov_max_dim: int = min(max_iter, krylov_restart, len(x0))


