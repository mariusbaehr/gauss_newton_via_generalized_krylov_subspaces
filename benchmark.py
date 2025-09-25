import timeit
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import gauss_newton
from gauss_newton_krylov import gauss_newton_krylov


def benchmark(res, x0, jac, error, kwargs={}, additional_methods=[], title=None):
    """
    For fast but still somewhat flexible benchmarks

    Parameters
    ----------
    res:
    x0:
    jac:
    error:
    args: Additional keyword arguments for gauss_newton


    """
    # TODO: make error optional as it is not always available

    def loss(x):
        return np.sum(res(x) ** 2)

    def callback(x, nfev):
        global error_list, loss_list, nfev_list
        error_list.append(error(x))
        loss_list.append(loss(x))
        nfev_list.append(nfev)

    def callback_scipy(intermediate_result):
        global error_list, loss_list, nfev_list
        error_list.append(error(intermediate_result.x))
        loss_list.append(loss(intermediate_result.x))
        nfev_list.append(intermediate_result.nfev)
        # print(intermediate_result.njev)

    def gn(callback):
        return gauss_newton(res, x0, jac, callback=callback, **kwargs)

    def ref_method(callback):
        return scipy.optimize.least_squares(res, x0, jac, callback=callback)

    def gnk(callback):
        return gauss_newton_krylov(res, x0, jac, callback=callback, **kwargs)

    def gnk_restart(callback):
        return gauss_newton_krylov(
            res, x0, jac, callback=callback, krylov_restart=20, **kwargs
        )

    methods = [ref_method, gnk, gnk_restart, gn] + additional_methods

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].set_title("Error Plot")
    ax[1].set_title("Loss Plot")
    timeit_number = 1
    for method in methods:
        global error_list, loss_list, nfev_list
        error_list = []
        loss_list = []
        nfev_list = []
        if method.__name__ in ["ref_method", "ref_cg"]:
            time = (
                timeit.timeit(lambda: method(None), number=timeit_number)
                / timeit_number
            )
            method(callback_scipy)
        else:
            time = (
                timeit.timeit(lambda: method(lambda: None), number=timeit_number)
                / timeit_number
            )
            method(callback)

        print(f"method = {method.__name__} time = {time} ")

        nit = len(error_list)  # actually this is nit+1
        ax[0].semilogy(range(nit), error_list, "x-", label=method.__name__)
        ax[1].semilogy(range(nit), loss_list, "x-", label=method.__name__)
        ax[2].plot(range(nit), nfev_list, "x-", label=method.__name__)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    if not title == None:
        plt.savefig(title + ".png", bbox_inches="tight")
    plt.show()
