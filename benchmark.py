import numpy as np
import scipy.optimize
from typing import List


def ref_method(res, x0, jac, args, callback, **kwargs):
    def cb_scipy(intermediate_result):
        callback(intermediate_result.x, intermediate_result.nfev, None)

    scipy.optimize.least_squares(res, x0, jac, callback=cb_scipy, args=args)

def reverse_accumulation(nfev_list: List[int]) -> List[int]:
    """
    The callback only gets the total, i.e. the accumulated count of residual evaluations,
    however for our plots we wish to plot only the number of evaluations per step.
    This funciton undoes that accumulation.

    """
    if not nfev_list:
        return []
    
    reversed_list = [nfev_list[0]] + [
        nfev_list[i] - nfev_list[i - 1] for i in range(1, len(nfev_list))
    ]
    return reversed_list




def benchmark_method(method, res, x0, jac, error, args=(), kwargs={}):

    def loss(x):
        return 0.5 * np.sum(res(x, *args) ** 2)

    error_list = [error(x0)]
    loss_list = [loss(x0)]
    nfev_list = []
    cg_iter_list = []

    def callback(x, nfev, cg_iter):
        error_list.append(error(x))
        loss_list.append(loss(x))
        if nfev is not None:
            nfev_list.append(nfev)
        if cg_iter is not None:
            cg_iter_list.append(cg_iter)

    method(res, x0, jac, args=args, callback=callback, **kwargs)

    nfev_list = reverse_accumulation(nfev_list)

    return error_list, loss_list, nfev_list, cg_iter_list
