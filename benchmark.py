import timeit
import scipy.optimize
import matplotlib.pyplot as plt
import numpy as np
from gauss_newton import gauss_newton
from gauss_newton_krylow import gauss_newton_krylow


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
    
    if "args" in kwargs:
        def loss(x):
            return np.sum(res(x, *kwargs["args"]) ** 2)
    else:
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

    def callback_cg(x):
        global error_list, loss_list, nfev_list
        error_list.append(error(x))
        loss_list.append(loss(x))
        nfev_list.append(0)

    def gn(callback):
        return gauss_newton(res, x0, jac, callback=callback, **kwargs)

    if "args" in kwargs:
        def ref_method(callback):
            return scipy.optimize.least_squares(res, x0, jac, callback=callback, args=kwargs["args"])
    else: 
        def ref_method(callback):
            return scipy.optimize.least_squares(res, x0, jac, callback=callback)

    def gnk(callback):
        return gauss_newton_krylow(res, x0, jac, callback=callback, **kwargs)

    def gnk_new_res(callback):
        return gauss_newton_krylow(
            res, x0, jac, callback=callback, version="res_new", **kwargs
        )

    def gnk_restart_new(callback):
        return gauss_newton_krylow(
            res,
            x0,
            jac,
            callback=callback,
            krylow_restart=20,
            version="res_new",
            **kwargs,
        )

    methods = [ref_method, gnk, gnk_restart_new, gn, gnk_new_res] + additional_methods
    # methods = [ref_method, gnk, gn, gnk_new_res] + additional_methods

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Iterationen")
    ax1.set_ylabel(r"Fehler $\log\|x_k-x^\ast\|$")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.set_xlabel("Iterationen")
    ax2.set_ylabel(r"Verlust $\log\mathcal{L}$")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.set_xlabel("Iterationen")
    ax3.set_ylabel(r"Anzahl $f$ Auswertungen")
    #    ax1.set_title("Error Plot")
    #    ax2.set_title("Loss Plot")
    #    ax3.set_title("fnev Plot")
    timeit_number = 1
    for method in methods:
        global error_list, loss_list, nfev_list
        error_list = [error(x0)]
        loss_list = [loss(x0)]
        nfev_list = [0]
        if method.__name__ == "ref_method":
            time = (
                timeit.timeit(lambda: method(None), number=timeit_number)
                / timeit_number
            )
            method(callback_scipy)
        elif method.__name__ == "ref_cg":
            method(callback_cg)
            time = None
        else:
            time = (
                timeit.timeit(lambda: method(lambda: None), number=timeit_number)
                / timeit_number
            )
            method(callback)

        print(f"method = {method.__name__} time = {time} ")

        nit = len(error_list)
        ax1.semilogy(range(nit), error_list, "x-", label=method.__name__)
        ax2.semilogy(range(nit), loss_list, "x-", label=method.__name__)
        ax3.plot(range(nit), nfev_list, "x-", label=method.__name__)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    if not title == None:
        fig1.savefig(title + "error" + ".png", bbox_inches="tight")
        fig2.savefig(title + "loss" + ".png", bbox_inches="tight")
        fig3.savefig(title + "nfev" + ".png", bbox_inches="tight")

    plt.show()
