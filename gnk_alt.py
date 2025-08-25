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
    beta_coordinate: npt.NDArray
    active_columns: int
    max_dim: int
    args: Tuple
    res: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray]
    jac: Callable[[npt.NDArray,Tuple[Any]],npt.NDArray]

    def __init__(self,res, beta0, jac, max_dim):
        if np.allclose(beta0,np.zeros_like(beta0)):
            raise ValueError("beta0 is not allowed to be 0 in the gauss_newton_krylov algorithm")

        self._res = res
        self._jac = jac
        self.active_columns = 1
        self.max_dim = max_dim
        

        self.beta_coordinate = np.zeros((self.max_dim))
        self.base = np.zeros((len(beta0),self.max_dim))

        self.beta_coordinate[0] = np.linalg.norm(beta0)
        self.base[:,0] =  beta0/self.beta_coordinate[0]

    @property
    def beta(self):
        return self.base[:,:self.active_columns]@self.beta_coordinate

    def update(self):

        self.normal_res -= self.base[:,:self.active_columns]@(self.base[:,:self.active_columns].T@self.normal_res) # shape (P,A)@(A,P)@P
        
        if np.allclose(self.normal_res,np.zeros_like(self.normal_res)):
            return None

        self.normal_res /= np.linalg.norm(self.normal_res)

        self.base[:,self.active_columns] = self.normal_res
        self.active_columns += 1
# an den neustart denken!!!!
# 
# !!!!!!!!!!!!!!!!!!!!!!




    def res(self, beta_coordinate, args):
        self.res_ev = self._res(self.base[:,:self.active_columns]@beta_coordinate[:self.active_columns], *args)
        return self.res_ev 

    def jac(self, beta_coordinate, args):
        jac_ev = self._jac(self.base[:,:self.active_columns]@beta_coordinate[:self.active_columns], *args)
        self.normal_res = jac_ev.T@self.res_ev
        return jac_ev@self.base[:,:self.active_columns]


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

    #count_sampels = len(res(beta,*args))
    count_parameters = len(beta0)

    if krylov_restart == None: 
        krylov_restart = max_iter

    max_dim = min(max_iter, krylov_restart, count_parameters) # Outside of Krylov, because could be useful for jacobian in future

    #jac_ev = np.zeros((count_sampels,max_dim))
    # Currently I don't see how one could implement this with jac(beta,v) which only would need to calculate a small number (active_colums) of directional derivatives, but to update the generalized krylov subspaces, the full jacobian is needed, which seems like a poor trade of for me

    krylov = Krylov(res, beta0.copy(), jac, max_dim)


    for iter in range(max_iter):

        res_ev: npt.NDArray = krylov.res(krylov.beta_coordinate,args)
        jac_ev: Union[npt.NDArray,sp.spmatrix] = krylov.jac(krylov.beta_coordinate,args)
        krylov.update()

        descent_direction, _, rank_jac, _ = scipy.linalg.lstsq(-1*jac_ev, res_ev) #Currently I'm operating under the assumption that the krylov base will typically be dense


        step_length: float = armijo_goldstein(krylov.res, krylov.beta_coordinate , jac_ev, args, descent_direction)

        
        squared_sum_beta_prev = np.sum(krylov.beta_coordinate**2)

        krylov.beta_coordinate[:krylov.active_columns] += step_length*descent_direction

        call_callback(callback,**{"beta":krylov.beta_coordinate,"iter":iter,"jac":jac,"rank_jac":rank_jac,"step_length":step_length})

        if step_length**2*np.sum(descent_direction**2) <= tol**2*squared_sum_beta_prev:
            success = True
            break


    if not success: print("Warning: The gauss_newton_krylov algorithm reached maximal iteration bound before terminating!")

    return RegressionResult("gauss newton krylov", krylov.beta, success, None, iter, None, iter)

