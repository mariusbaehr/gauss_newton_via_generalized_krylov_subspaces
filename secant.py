from typing import List, Union
import numpy.typing as npt
import scipy.sparse as sp
from gauss_newton_krylov import Krylov

class Secant:
    """

    Attributes
    ----------

    """

    jac: Union[npt.NDArray, sp.spmatrix]
    delta_res: List[npt.NDArray] # maybe store this in a matrice
    delta_x: List[npt.NDArray]
    secant_update: int
    krylov: Union[None,Krylov]

    def __init__(self, jac: Union[npt.NDArray, sp.spmatrix], delta_res: List[npt.NDArray], delta_x: List[npt.NDArray], secant_update: int, krylov: Union[None,Krylov]):
        self.jac = jac
        self.delta_res = delta_res
        self.delta_x = delta_x
        self.secant_update = secant_update
        self.krylov= krylov

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!

# NOTE: If secant_update > P then this is stupid with lists, but if secant_update<<P its very smart

# !!!!!!!!!!!!!!!!!!!!!

    def update(self):

    def evaluate(self):





