from collections.abc import Callable
from typing import Union, Any, Tuple
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy


def linear_least_squares(A: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """
    Calculates least squares solution of linear problem e.g. ||y-Ax||.

    Parameters
    ----------
    A: Matrice.
    y: Vector.

    Returns
    -------
    x: Linear least squares solution.
    """

    q, r = scipy.linalg.qr(A, mode="economic")

    for r_kk in np.diagonal(r):
        if np.isclose(r_kk, 0, atol=1e-8):
            print("A is rank deficient")
    x = scipy.linalg.solve_triangular(r, q.T @ y)
    return x


def modified_gram_schmidt(
    basis: npt.NDArray, vector: npt.NDArray, atol=1e-8
) -> npt.NDArray:  # TODO: Add tol
    """
    Orthonormalize vector with respect to orthonormal basis.

    Parameters
    ----------
    basis: Basis vectores are stored as colums.
    vector: Vector which should be orthogonalized.
    atol: If ||vector||<atol, vector is considert linear depended of basis, hence method will stop there and raise GeneralizedKrylowSubspaceBreakdown Exception.

    Returns
    -------
    vector: Orthonormalized vector.
    """
    for column in basis.T:
        column_dot_vector = np.dot(column, vector)

        vector -= (column_dot_vector) * column

    vector_norm = np.linalg.norm(vector)
    if np.isclose(vector_norm, 0, atol=atol, rtol=0):
        raise GeneralizedKrylowSubspaceBreakdown(
            "Normal residual is allready inside generalized Krylow Subspcae, there for gauss newton krylow algorithm has to proceed without enlarging subspace."
        )

    vector /= vector_norm

    return vector


class GeneralizedKrylowSubspaceBreakdown(Exception):
    pass  # TODO: Not shure if Error is apropriate, as it could naturaly happen
    # TODO: Maybe add error message here


class GeneralizedKrylowSubspaceSpansEntireSpace(Exception):
    pass


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
        self.basis = (x0 / x0_norm).reshape(-1, 1)
        x_coordinate = np.array([x0_norm])
        return x_coordinate

    def x(self, x_coordinate: npt.NDArray) -> npt.NDArray:
        return self.basis @ x_coordinate

    def evaluate(
        self,
        fun: Callable[[npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix]],
        x_coordinate: npt.NDArray,
        *args: Any,
    ) -> Union[npt.NDArray, sp.spmatrix]:
        """
        For evaluating functions such as res or jac on the generalized krylow subspace.
        """
        return fun(self.x(x_coordinate), *args)

    def update(
        self, jac_ev: Union[npt.NDArray, sp.spmatrix], res_ev: npt.NDArray
    ) -> None:

        if self.basis.shape[0] == self.basis.shape[1]:
            raise GeneralizedKrylowSubspaceSpansEntireSpace

        normal_res = -jac_ev.T @ res_ev

        try:
            normal_res = modified_gram_schmidt(
                self.basis, normal_res
            )  # TODO: Determine if GKS breaks down based on modified gram schmidt process
        except GeneralizedKrylowSubspaceBreakdown:
            raise  # Exception will just be passed to gauss_newton_krylow

        normal_res /= np.linalg.norm(normal_res)
        normal_res = normal_res.reshape(-1, 1)
        self.basis = np.hstack([self.basis, normal_res])


def gauss_newton_krylow(
    res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
    x0: npt.NDArray,
    jac: Callable[[npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix]],
    krylow_restart: int | None = None,
    args: Tuple = (),
    tol: float = 1e-8,
    max_iter=100,
    callback: Callable = lambda: None,
    version: str = "res_old",
) -> RegressionResult:
    """
    Parameters
    ----------
    res: The residual function, called as res(x, *args) the argument x and its return must always be ndarrays.
    x0: Initial guess of the regression parameters.
    jac: Jacobian of residual function, called as jac(x, *args).
    krylow_restart: If the krylow_basis gets larger than krylow_restart the basis is reset, if left to None basis never resets.
    args: Additional arguments passed to res and jac.
    tol: Tolerance for termination by the change of the paramters x.
    max_iter: Maximum number of iterations.
    callback: Called as callback with the following possible positional arguments: x, iter, jac, rank_jac, step_length, nfev, cg_iter. Implementet arguments are automaticaly determined by call_callback. cg_iter was only added to be consitent with gauss_newton callback.
    version: Must be one of ['res_old','res_new','jac_old_res_old','jac_old_res_new']. The generalized Krylow supspace will be expanded with -jac(x_new).T@res(x_old), -jac(x_new).T@res(x_new), -jac(x_old).T@res(x_old), -jac(x_old).T@res(x_new). Defaults to 'res_old' as suggested in "An Efficient Implementation of the Gauss–Newton Method Via Generalized Krylow Subspaces", we recomend 'res_new' however.

    Returns
    -------
    res: Returns instance of RegressionResult.
    """

    success: bool = False

    krylow = GeneralizedKrylowSubspace()
    x_coordinate = krylow.start(x0)

    def res_krylow(x_coordinate, *args):
        return krylow.evaluate(res, x_coordinate, *args)

    res_ev_new: npt.NDArray = res_krylow(x_coordinate, *args)
    nfev: int = 1
    jac_ev: sp.spmatrix = jac(x0, *args)
    njev: int = 1  # TODO: Use this instead of iter

    if krylow_restart == None:
        krylow_restart = max_iter

    for iter in range(1, max_iter):

        jac_krylow = jac_ev @ krylow.basis
        res_ev = res_ev_new

        # descent_direction, _, _, _ = scipy.linalg.lstsq(-1 * jac_krylow, res_ev)
        descent_direction = linear_least_squares(-1 * jac_krylow, res_ev)

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
                "cg_iter": None,
            },
        )

        if step_length**2 * np.sum(descent_direction**2) <= tol**2 * squared_sum_x_prev:
            success = True
            break

        jac_ev_old = jac_ev
        jac_ev = krylow.evaluate(jac, x_coordinate, *args)

        try:
            if version == "res_old":
                krylow.update(jac_ev, res_ev)
            elif version == "res_new":
                krylow.update(jac_ev, res_ev_new)
            elif version == "jac_old_res_old":
                krylow.update(jac_ev_old, res_ev)
            elif version == "jac_old_res_new":
                krylow.update(jac_ev_old, res_ev_new)
            else:
                raise ValueError(
                    "Variable version must be in ['res_old','res_new','jac_old_res_old','jac_old_res_new']"
                )  # TODO Check before hand

            x_coordinate = np.append(x_coordinate, 0)
        except GeneralizedKrylowSubspaceBreakdown:  # TODO warnings
            print(
                f"Generalized krylow subspace breakdown at iteration = {iter}, basis.shape = {krylow.basis.shape}"
            )

        except GeneralizedKrylowSubspaceSpansEntireSpace:  # TODO warnings
            print(
                f"Warning: The genearlized krylow subspace is now identical to the whole parameter space at iteration = {iter}"
            )

        if iter % krylow_restart == 0:
            x_coordinate = krylow.start(krylow.x(x_coordinate))

    if not success:
        print(
            "Warning: The gauss_newton_krylow algorithm reached maximal iteration bound before terminating!"
        )

    return RegressionResult(
        "gauss newton krylow", krylow.x(x_coordinate), success, nfev, iter, None, iter
    )
