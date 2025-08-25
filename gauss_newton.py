from collections.abc import Callable
from typing import Union, Any, Tuple 
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy


def gauss_newton(res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray], beta0: npt.NDArray, jac: Callable[[npt.NDArray,Tuple[Any]],Union[npt.NDArray, sp.spmatrix]], args: Tuple = (), tol: float=1E-8, max_iter=100, step_length_control: Callable = armijo_goldstein, callback: Callable=lambda : None) -> RegressionResult : 
    """
    Gauss Newton algorithm for minimizing ||res(theta)|| with respect to theta.

    Parameters
    ----------

    res: The residual function, called as res(beta, *args) the argument beta and its return must always be ndarrays.
    beta0: Initial guess of the regression parameters.
    jac: Called as jac(beta, *args), should return either a NDArray or a spmatrix, if returned a spmatrix a sparse solver is used to solve the linearised equation.
    args: 
    tol:
    max_iter: 
    step_length_control: Only for demonstrational purposes
    callback: Called as callback with the following possible positional arguments: beta, iter, jac, rank_jac, step_length


    Returns
    -------
    res: RegressionResult
    
    """
    
    beta: npt.NDArray = beta0.copy()
    success: bool = False

    rank_jac: int | None = None # rank of jacobian might be interesting to inspect, but is only available for dense solver


    for iter in range(max_iter):

        res_ev: npt.NDArray = res(beta,*args)
        jac_ev: Union[npt.NDArray,sp.spmatrix] = jac(beta,*args) 
        is_sparse: bool = isinstance(jac_ev,(sp.sparray,sp.spmatrix)) #TODO are there other sp types?

        if is_sparse:
            descent_direction, *_ = scipy.sparse.linalg.lsqr(-1*jac_ev, res_ev)
        else:
            descent_direction, _, rank_jac, _ = scipy.linalg.lstsq(-1*jac_ev, res_ev)

        #TODO: Hand over res_ev to step_length_control and also get the latest back to save two function evaluations per iteration
        #Note if signature changed all test must be changed too
        #TODO: Also the nrev used
        step_length: float = step_length_control(res, beta, jac_ev, args, descent_direction)
        #step_length, res_ev: Tuple[float,npt.NDArray] = step_length_control(res,beta,res_ev, jac_ev,args,descent_direction)

        
        squared_sum_beta_prev = np.sum(beta**2)

        beta += step_length*descent_direction

        call_callback(callback,**{"beta":beta,"iter":iter,"jac":jac,"rank_jac":rank_jac,"step_length":step_length})

        if step_length**2*np.sum(descent_direction**2) <= tol**2*squared_sum_beta_prev:
            success = True
            break


    if not success: print("Warning: The gauss_newton algorithm reached maximal iteration bound before terminating!")

    return RegressionResult("gauss newton", beta, success, None, iter, None, iter)



