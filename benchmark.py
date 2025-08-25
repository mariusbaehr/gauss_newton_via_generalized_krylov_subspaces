import timeit
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import gauss_newton
from gauss_newton_krylov import gauss_newton_krylov

def benchmark(res, beta0, jac, error, kwargs = {}, additional_methods=[]):
    """
    For fast but still somewhat flexible benchmarks

    Parameters
    ----------
    res:
    beta0:
    jac:
    error:
    args: Additional keyword arguments for gauss_newton


    """
    #TODO: make error optional as it is not always available

    def loss(beta): return np.sum(res(beta)**2)

    def cb_error_loss(beta):
        global error_list, loss_list
        error_list.append(error(beta))
        loss_list.append(loss(beta))

    def cb_error_loss_scipy(xk):
        global error_list, loss_list
        error_list.append(error(xk))
        loss_list.append(loss(xk))

    def gn(callback):
        return gauss_newton(res,beta0,jac,callback=callback,**kwargs)
    def ref_method(callback):
        return scipy.optimize.least_squares(res,beta0,jac,callback=callback)
    def gnk(callback):
        return gauss_newton_krylov(res,beta0,jac,callback=callback,**kwargs)
    def gnk_restart(callback):
        return gauss_newton_krylov(res,beta0,jac,callback=callback, krylov_restart=20,**kwargs)
        

    methods = [gn,ref_method,gnk,gnk_restart] + additional_methods


    fig, ax = plt.subplots(2,1, figsize=(10,10))
    ax[0].set_title("Error Plot")
    ax[1].set_title("Loss Plot")
    timeit_number= 1
    for method in methods:
        global error_list, loss_list
        error_list = []
        loss_list = []
        if method.__name__ in ["ref_method", "ref_cg"]: 
            time = timeit.timeit(lambda:method(None),number=timeit_number)/timeit_number
            method(cb_error_loss_scipy) 
        else: 
            time = timeit.timeit(lambda:method(lambda : None),number=timeit_number)/timeit_number
            method(cb_error_loss)

        print(f"method = {method.__name__} time = {time} ")

        nit = len(error_list) # actually this is nit+1
        ax[0].semilogy(range(nit),error_list, "x-", label=method.__name__)
        ax[1].semilogy(range(nit),loss_list, "x-", label=method.__name__)


    ax[0].legend()
    ax[1].legend()
    plt.show()

