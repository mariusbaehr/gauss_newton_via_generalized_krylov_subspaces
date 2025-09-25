from collections.abc import Callable
from typing import Union, Any, Tuple
from regression_result import RegressionResult
from armijo_goldstein import armijo_goldstein
from call_callback import call_callback
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy


class Krylov:
    """
    The goal of this class is to outsource all the additional complexity from gauss_newton_krylov. To be close to the implementation of gauss_newton.

    Attributes
    ----------
    base:
    active_columns:
    _res:
    _jac:
    krylov_max_dim:
    res_ev:
    restart:
    """

    base: npt.NDArray
    active_columns: int
    _res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray]
    _jac: Callable[
        [npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix, sp.sparray]
    ]
    krylov_max_dim: int
    res_ev: npt.NDArray
    restart: bool

    def __init__(
        self,
        _res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
        x0: npt.NDArray,
        _jac: Callable[
            [npt.NDArray, Tuple[Any]], Tuple[npt.NDArray, sp.spmatrix, sp.sparray]
        ],
        krylov_max_dim: int,
        restart: bool,
        *args
    ):

        if np.allclose(x0, np.zeros_like(x0)):
            raise ValueError(
                "x0 is not allowed to be 0 in the gauss_newton_krylov algorithm"
            )

        self._res = _res
        self._jac = _jac
        self.active_columns = 1
        self.krylov_max_dim = krylov_max_dim
        self.restart = restart

        self.base = np.zeros((len(x0), self.krylov_max_dim))
        self.base[:, 0] = x0 / np.linalg.norm(x0)

        self.jac_ev = self._jac(
            self.base[:, : self.active_columns] @ np.array([np.linalg.norm(x0)]), *args
        )

    def x(self, x_coordinate: npt.NDArray):
        return self.base[:, : self.active_columns] @ x_coordinate

    def res(self, x_coordinate: npt.NDArray, *args: Any) -> npt.NDArray:
        return self._res(self.base[:, : self.active_columns] @ x_coordinate, *args)

    # def jac(self, x_coordinate: npt.NDArray, *args:Any) -> npt.NDArray:
    #    jac_ev = self._jac(self.base[:,:self.active_columns]@x_coordinate, *args)
    #    self.normal_res = jac_ev.T@self.res_ev
    #    return jac_ev@self.base[:,:self.active_columns]

    def jac(self) -> npt.NDArray:
        # Die Krylov auswertung wird so versteckt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return self.jac_ev @ self.base[:, : self.active_columns]

    def update(
        self, x_coordinate: npt.NDArray, res_ev: npt.NDArray, *args: Tuple[Any]
    ) -> npt.NDArray:
        """
        Because the dimensions may change x_coordinate is also updatet in size.
        """

        np.testing.assert_allclose(self.base[:,:self.active_columns].T @ self.base[:,:self.active_columns], np.eye(self.active_columns), atol=1E-8, rtol=0, err_msg=f"first violation = {self.active_columns}")

        self.jac_ev = self._jac(
            self.base[:, : self.active_columns] @ x_coordinate, *args
        )

        print(self.base[:,:self.active_columns].T @ self.base[:,:self.active_columns])

        normal_res = self.jac_ev.T @ res_ev
        if self.active_columns == self.krylov_max_dim and not self.restart:
            print(
                "Warning: gauss_newton_krylov algorithm reached krylov_max_dim. Further iterations are equivalent to the standart gauss_newton algorithm but much slower."
            )

            # np.testing.assert_allclose(self.base.T @ self.base, np.eye(len(self.base)))

            return x_coordinate


        normal_res -= self.base[:, : self.active_columns] @ (
            self.base[:, : self.active_columns].T @ normal_res
        )

        if np.allclose(normal_res, np.zeros_like(normal_res)):
            return x_coordinate

        normal_res /= np.linalg.norm(normal_res)

        if self.active_columns == self.krylov_max_dim and self.restart:
            x0 = self.x(x_coordinate)
            x_coordinate = np.array([np.linalg.norm(x0)])
            self.base = np.zeros((len(normal_res), self.krylov_max_dim))
            self.base[:, 0] = x0 / x_coordinate[0]
            self.active_columns = 1
            # TODO: Seperate the krylov update and the restart formular (But now its weird)
        else:

            self.base[:, self.active_columns] = normal_res
            self.active_columns += 1
            x_coordinate = np.block([x_coordinate, np.zeros(1)])

        return x_coordinate


def gauss_newton_krylov(
    res: Callable[[npt.NDArray, Tuple[Any]], npt.NDArray],
    x0: npt.NDArray,
    jac: Callable[
        [npt.NDArray, Tuple[Any]], Union[npt.NDArray, sp.spmatrix, sp.sparray]
    ],
    krylov_restart: int | None = None,
    args: Tuple = (),
    tol: float = 1e-8,
    max_iter=100,
    callback: Callable = lambda: None,
) -> RegressionResult:
    """
    Parameters
    ----------
    res: The residual function, called as res(x, *args) the argument x and its return must always be ndarrays.
    x0: Initial guess of the regression parameters.
    jac: Called as jac(x,d, *args), should return either a NDArray or a spmatrix, if returned a spmatrix a sparse solver is used to solve the linearised equation.
    krylov_restart: If the krylov_base gets larger than krylov_restart the base is reset, if left to None base never resets.
    args: Additional arguments passed to res and jac.
    tol: Tolerance for termination by the change of the paramters x.
    max_iter: Maximum number of iterations.
    callback: Called as callback with the following possible positional arguments: x, iter, jac, rank_jac, step_length, nfev. Implementet arguments are automaticaly determined by call_callback.

    Returns
    -------
    res: Returns instance of RegressionResult.
    """

    success: bool = False
    rank_jac: int | None = None

    restart: bool = True
    if krylov_restart == None:
        krylov_restart = max_iter
        restart = False

    krylov_max_dim: int = min(max_iter, krylov_restart, len(x0))
    x_coordinate = np.array([np.linalg.norm(x0)])

    krylov = Krylov(res, x0.copy(), jac, krylov_max_dim, restart, *args)

    res_ev: npt.NDArray = krylov.res(x_coordinate, *args)
    nfev: int = 1

    for iter in range(max_iter):

        jac_ev: npt.NDArray = krylov.jac()

        descent_direction, _, rank_jac, _ = scipy.linalg.lstsq(
            -1 * jac_ev, res_ev
        )  # I'm operating under the assumption that the krylov base will typically be dense

        step_length, res_ev, nfev_delta = armijo_goldstein(
            krylov.res, x_coordinate, res_ev, jac_ev, args, descent_direction
        )
        nfev += nfev_delta

        squared_sum_x_prev = np.sum(x_coordinate**2)

        x_coordinate += step_length * descent_direction

        call_callback(
            callback,
            **{
                "x": krylov.x(x_coordinate),
                "iter": iter,
                "jac": jac,
                "rank_jac": rank_jac,
                "step_length": step_length,
                "nfev": nfev,
            }
        )

        if step_length**2 * np.sum(descent_direction**2) <= tol**2 * squared_sum_x_prev:
            success = True
            break

        x_coordinate = krylov.update(x_coordinate, res_ev, *args)

    if not success:
        print(
            "Warning: The gauss_newton_krylov algorithm reached maximal iteration bound before terminating!"
        )

    return RegressionResult(
        "gauss newton krylov", krylov.x(x_coordinate), success, nfev, iter, None, iter
    )
