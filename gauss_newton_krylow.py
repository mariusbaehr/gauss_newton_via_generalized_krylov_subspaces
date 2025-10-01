from collections.abc import Callable
from typing import Union, Any, Tuple
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy

def modified_gram_schmidt(basis: npt.NDArray, vector: npt.NDArray)-> npt.NDArray:
    for column in basis.T:
        vector -= (column @ vector) * column
    return vector


class GeneralizedKrylowSubspaceBreakdown(Exception):
    pass # TODO: Not shure if Error is apropriate, as it could naturaly happen
        # TODO: Maybe add error message here


class GeneralizedKrylowSubspace:
    """
    The goal of this class is to outsource complexity from gauss_newton_krylow. To be closer to the implementation of gauss_newton.

    Attributes
    ----------
    basis: The basis of the genearalized Krylow subspace, represented as a matrix with basis vectors as columns.
    """

    basis: npt.NDArray

    def __init__(self):
        pass

    def start(self, x0: npt.NDArray) -> npt.NDArray:
        if np.allclose(x0, np.zeros_like(x0)):
            raise ValueError(
                "x0 is not allowed to be 0 in the gauss_newton_krylow algorithm"
            )

        x0_norm = np.linalg.norm(x0)
        self.basis = (x0 / x0_norm).reshape(-1,1)
        x_coordinate = np.array([x0_norm])
        return x_coordinate

    def x(self, x_coordinate: npt.NDArray)->npt.NDArray:
        return self.basis@ x_coordinate

    def evaluate(self, fun:Callable[ [npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix] ], x_coordinate: npt.NDArray,*args: Any)->Union[npt.NDArray, sp.spmatrix]:
        """
        For evaluating functions such as res or jac on the generalized krylow subspace.
        """
        return fun(self.x(x_coordinate),*args)

    def update(
        self, jac_ev: Union[npt.NDArray,sp.spmatrix], res_ev: npt.NDArray
    ) -> None:
        normal_res = jac_ev.T @ res_ev #TODO: Check sign
        #normal_res -= self.basis @ ( self.basis.T @ normal_res)
        normal_res = modified_gram_schmidt(self.basis, normal_res)

        if np.allclose(normal_res, np.zeros_like(normal_res)):
            raise GeneralizedKrylowSubspaceBreakdown(
                "Normal residual is allready inside generalized Krylow Subspcae, there for gauss newton krylow algorithm has to proceed without enlarging subspace."
            )

        normal_res /= np.linalg.norm(normal_res)
        normal_res = normal_res.reshape(-1,1)
        self.basis = np.hstack([self.basis, normal_res])
    

def gauss_newton_krylow(
    res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
    x0: npt.NDArray,
    jac: Callable[
        [npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix]
    ],
    krylow_restart: int | None = None,
    args: Tuple = (),
    tol: float = 1e-8,
    max_iter=100,
    callback: Callable = lambda: None,
    version: str = "res_old"
) -> RegressionResult:
    """
    Parameters
    ----------
    res: The residual function, called as res(x, *args) the argument x and its return must always be ndarrays.
    x0: Initial guess of the regression parameters.
    jac: 
    krylow_restart: If the krylow_basis gets larger than krylow_restart the basis is reset, if left to None basis never resets.
    args: Additional arguments passed to res and jac.
    tol: Tolerance for termination by the change of the paramters x.
    max_iter: Maximum number of iterations.
    callback: Called as callback with the following possible positional arguments: x, iter, jac, rank_jac, step_length, nfev. Implementet arguments are automaticaly determined by call_callback.

    Returns
    -------
    res: Returns instance of RegressionResult.
    """

    success: bool = False

    krylow = GeneralizedKrylowSubspace()
    x_coordinate = krylow.start(x0)

    def res_krylow(x_coordinate,*args):
        return krylow.evaluate(res,x_coordinate,*args)

    res_ev_new: npt.NDArray = res_krylow(x_coordinate,*args)
    nfev: int = 1
    jac_ev: sp.spmatrix = jac(x0,*args)
    njev: int = 1 #TODO: Use this instead of iter

    if krylow_restart == None:
        krylow_restart = max_iter

    for iter in range(1,max_iter):

        jac_krylow = jac_ev @ krylow.basis
        res_ev = res_ev_new

        descent_direction, _, _, _ = scipy.linalg.lstsq(
            -1 * jac_krylow, res_ev 
        )

        step_length, res_ev_new, nfev_delta = armijo_goldstein(
            res_krylow, x_coordinate, res_ev, jac_krylow, args, descent_direction
        )
        nfev += nfev_delta

        squared_sum_x_prev = np.sum(x_coordinate**2)

        x_coordinate += step_length * descent_direction

        call_callback(
            callback,
            **{
                "x": krylow.x(x_coordinate),
                "iter": iter,
                "jac": jac,
                "step_length": step_length,
                "nfev": nfev,
            }
        )

        if step_length**2 * np.sum(descent_direction**2) <= tol**2 * squared_sum_x_prev:
            success = True
            break

        jac_ev = krylow.evaluate(jac, x_coordinate)

        if iter % krylow_restart == 0: #TODO it might be more reasonable to restart based on krylow.basis dimension
            x_coordinate = krylow.start(krylow.x(x_coordinate))
        try: 
            if version == "res_old":
                krylow.update(jac_ev,res_ev)
            elif version == "res_new":
                krylow.update(jac_ev,res_ev_new)
            else:
                raise ValueError("Variable version must be in ['res_old','res_new']") # TODO Check before hand


            x_coordinate = np.append(x_coordinate,0)
        except GeneralizedKrylowSubspaceBreakdown:
            pass



    if not success:
        print(
            "Warning: The gauss_newton_krylow algorithm reached maximal iteration bound before terminating!"
        )

    return RegressionResult(
        "gauss newton krylow", krylow.x(x_coordinate), success, nfev, iter, None, iter
    )
