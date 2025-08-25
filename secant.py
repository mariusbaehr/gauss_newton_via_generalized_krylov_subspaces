from typing import List, Union
import numpy.typing as npt
import scipy.sparse as sp
from gauss_newton_krylov import KrylovBase

class Secant:
    """

    Attributes
    ----------

    """

    jac: Union[npt.NDArray, sp.spmatrix]
    delta_res: List[npt.NDArray] # maybe store this in a matrice
    delta_beta: List[npt.NDArray]
    secant_update: int
    krylov_base: Union[None,KrylovBase]

    def __init__(self, jac: Union[npt.NDArray, sp.spmatrix], delta_res: List[npt.NDArray], delta_beta: List[npt.NDArray], secant_update: int, krylov_base: Union[None,KrylovBase]):
        self.jac = jac
        self.delta_res = delta_res
        self.delta_beta = delta_beta
        self.secant_update = secant_update
        self.krylov_base = krylov_base

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# NOTE: If secant_update > P then this is stupid with lists, but if secant_update<<P its very smart

# !!!!!!!!!!!!!!!!!!!!!

    def update(self):

    def evaluate(self):





