from collections.abc import Callable
from typing import Union, Any, Tuple
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

class GeneralizedKrylowSubspaceBreakdown(Exception):
    pass


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
            normal_res = modified_gram_schmidt(self.basis, normal_res)
        except GeneralizedKrylowSubspaceBreakdown:
            raise  # Exception will just be passed to gauss_newton_krylow

        normal_res /= np.linalg.norm(normal_res)
        normal_res = normal_res.reshape(-1, 1)
        self.basis = np.hstack([self.basis, normal_res])
