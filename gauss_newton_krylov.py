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


    """
    base: npt.NDArray
    active_columns: int
    _res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray]
    _jac: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray]
    krylov_max_dim: int
    res_ev: npt.NDArray
    restart: bool

    def __init__(self, _res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray], beta0: npt.NDArray, _jac: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray], krylov_max_dim: int, restart: bool, *args):

        if np.allclose(beta0,np.zeros_like(beta0)):
            raise ValueError("beta0 is not allowed to be 0 in the gauss_newton_krylov algorithm")

        self._res = _res
        self._jac = _jac
        self.active_columns = 1
        self.krylov_max_dim = krylov_max_dim
        self.restart = restart

        self.base = np.zeros((len(beta0),self.krylov_max_dim))
        self.base[:,0] = beta0/np.linalg.norm(beta0)

        self.jac_ev = self._jac(self.base[:,:self.active_columns]@np.array([np.linalg.norm(beta0)]), *args)

    def beta(self,beta_coordinate:npt.NDArray):
        return self.base[:,:self.active_columns]@beta_coordinate

    def res(self, beta_coordinate: npt.NDArray, *args:Any) -> npt.NDArray:
        return self._res(self.base[:,:self.active_columns]@beta_coordinate, *args)

    #def jac(self, beta_coordinate: npt.NDArray, *args:Any) -> npt.NDArray: 
    #    jac_ev = self._jac(self.base[:,:self.active_columns]@beta_coordinate, *args)
    #    self.normal_res = jac_ev.T@self.res_ev
    #    return jac_ev@self.base[:,:self.active_columns]

    def jac(self) -> npt.NDArray:
        return self.jac_ev@self.base[:,:self.active_columns]

    def update(self,beta_coordinate,res_ev,*args)->npt.NDArray:
        """
        Because the dimensions may change beta_coordinate is also updatet in size.


        """

        self.jac_ev = self._jac(self.base[:,:self.active_columns]@beta_coordinate, *args)
        #normal_res = self.jac_ev.T@res_ev
        normal_res = self.jac_ev.T@self.res(beta_coordinate,*args)

        normal_res -= self.base[:,:self.active_columns]@(self.base[:,:self.active_columns].T@normal_res)

        if np.allclose(normal_res,np.zeros_like(normal_res)):
            return beta_coordinate

        normal_res /= np.linalg.norm(normal_res)

        if self.active_columns == self.krylov_max_dim:# and self.restart:
            beta0=self.beta(beta_coordinate)
            beta_coordinate = np.array([np.linalg.norm(beta0)])
            self.base = np.zeros((len(normal_res),self.krylov_max_dim))
            self.base[:,0] = beta0/beta_coordinate[0]
            self.active_columns = 1
            #TODO: Seperate the krylov update and the restart formular
            #TODO: If no restart was set, just continue
        else:

            self.base[:,self.active_columns] = normal_res
            self.active_columns +=1
            beta_coordinate = np.block([beta_coordinate,np.zeros(1)])
        
        return beta_coordinate



def gauss_newton_krylov(
        res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray], 
        beta0: npt.NDArray, 
        #jac: Callable[[npt.NDArray,npt.NDArray,Tuple[Any]],npt.NDArray],
        jac: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray],
        krylov_restart: int | None = None,
        args: Tuple = (),
        tol: float=1E-8,
        max_iter=100,
        callback: Callable=lambda : None)->RegressionResult:
    """


    Parameters
    ----------

    res: The residual function, called as res(beta, *args) the argument beta and its return must always be ndarrays.
    beta0: Initial guess of the regression parameters.
    jac: Called as jac(beta,d, *args), should return either a NDArray or a spmatrix, if returned a spmatrix a sparse solver is used to solve the linearised equation.
    krylov_restart: 
    args: 
    tol:
    max_iter: 
    callback: Called as callback with the following possible positional arguments: beta, iter, jac, rank_jac, step_length


    Returns
    -------
    res: RegressionResult
    
    """

    success: bool = False
    rank_jac: int | None = None

    restart: bool = True
    if krylov_restart == None:
        krylov_restart = max_iter
        restart = False

    krylov_max_dim: int = min(max_iter, krylov_restart, len(beta0))
    beta_coordinate = np.array([np.linalg.norm(beta0)])

    krylov = Krylov(res, beta0.copy(), jac, krylov_max_dim,restart, *args)

    for iter in range(max_iter):

        res_ev: npt.NDArray = krylov.res(beta_coordinate,*args)
        #jac_ev: npt.NDArray = krylov.jac(beta_coordinate,*args) 
        jac_ev: npt.NDArray = krylov.jac()
        #Currently I'm operating under the assumption that the krylov base will typically be dense

        descent_direction, _, rank_jac, _ = scipy.linalg.lstsq(-1*jac_ev, res_ev)

        step_length: float = armijo_goldstein(krylov.res, beta_coordinate, jac_ev, args, descent_direction)


        squared_sum_beta_prev = np.sum(beta_coordinate**2)
        
        beta_coordinate += step_length*descent_direction

        call_callback(callback, **{"beta":krylov.beta(beta_coordinate),"iter":iter, "jac":jac, "rank_jac": rank_jac, "step_length":step_length})

        if step_length**2*np.sum(descent_direction**2) <= tol**2*squared_sum_beta_prev:
            success = True
            break

        beta_coordinate = krylov.update(beta_coordinate,res_ev, *args)

    if not success: print("Warning: The gauss_newton_krylov algorithm reached maximal iteration bound before terminating!")


    return RegressionResult("gauss newton krylov", krylov.beta(beta_coordinate), success, None, iter, None, iter)


    
