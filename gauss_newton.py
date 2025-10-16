from collections.abc import Callable
from typing import Union, Any, Tuple
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy


def cg_least_squares(
    A: sp.spmatrix, y: npt.NDArray, x0: npt.NDArray | None = None
) -> Tuple[npt.NDArray, int]:
    """ """
    p = A.shape[1]

    ATA = scipy.sparse.linalg.LinearOperator((p, p), matvec=lambda x: A.T @ (A @ x))

    global cg_iter
    cg_iter = 0

    def cb_iter(x):
        global cg_iter
        cg_iter += 1

    x, _ = scipy.sparse.linalg.cg(ATA, A.T @ y, x0=x0, callback=cb_iter)

    return x, cg_iter


def gauss_newton(
    res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
    x0: npt.NDArray,
    jac: Callable[[npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix]],
    args: Tuple = (),
    tol: float = 1e-8,
    max_iter=100,
    step_length_control: Callable = armijo_goldstein,
    callback: Callable = lambda: None,
) -> RegressionResult:
    """
    Gauss Newton algorithm for minimizing ||res(theta)|| with respect to theta.

    Parameters
    ----------

    res: The residual function, called as res(x, *args) the argument x and its return must always be ndarrays.
    x0: Initial guess of the regression parameters.
    jac: Called as jac(x, *args), should return either a NDArray or a spmatrix, if returned a spmatrix a sparse solver is used to solve the linearised equation.
    args:
    tol:
    max_iter:
    step_length_control: Only for demonstrational purposes
    callback: Called as callback with the following possible positional arguments: x, iter, jac, rank_jac, step_length


    Returns
    -------
    res: RegressionResult

    """

    x: npt.NDArray = x0.copy()
    success: bool = False

    cg_iter: int | None = None

    res_ev: npt.NDArray = res(x, *args)
    nfev: int = 1

    for iter in range(1, max_iter):

        # res_ev is updated in step_length control
        jac_ev: Union[npt.NDArray, sp.spmatrix] = jac(x, *args)
        is_sparse: bool = isinstance(jac_ev, (sp.sparray, sp.spmatrix))

        if is_sparse:
            # descent_direction, _, cg_iter, *_ = scipy.sparse.linalg.lsqr( -1.0 * jac_ev, res_ev)
            descent_direction, cg_iter = cg_least_squares(-jac_ev, res_ev, x)
        else:
            descent_direction, _, _, _ = scipy.linalg.lstsq(-1 * jac_ev, res_ev)

        step_length, res_ev, nfev_delta = step_length_control(
            res, x, res_ev, jac_ev, args, descent_direction
        )
        nfev += nfev_delta

        squared_sum_x_prev = np.sum(x**2)

        x += step_length * descent_direction

        call_callback(
            callback,
            **{
                "x": x,
                "iter": iter,
                "jac": jac,
                "step_length": step_length,
                "nfev": nfev,
                "cg_iter": cg_iter,
            }
        )

        if step_length**2 * np.sum(descent_direction**2) <= tol**2 * squared_sum_x_prev:
            success = True
            break

    if not success:
        print(
            "Warning: The gauss_newton algorithm reached maximal iteration bound before terminating!"
        )

    return RegressionResult("gauss newton", x, success, nfev, iter, None, iter)
