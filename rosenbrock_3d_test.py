import numpy as np
import scipy.sparse 
import matplotlib.pyplot as plt
from matplotlib import cm
from gauss_newton import gauss_newton

P = 2

def res(beta):
    block1 = 10*(beta[1:]-beta[:-1]**2)
    block2 = 1-beta[:-1]
    return np.concatenate([block1,block2])

beta0 = np.array([-1.0,1.0])

def jac(beta):
    block1 = 10*scipy.sparse.eye(P-1,P,k=1) - 20*scipy.sparse.diags(beta[:-1],shape=(P-1,P))
    block2 = -scipy.sparse.eye(P-1,P,k=0)
    return scipy.sparse.block_array([[block1],[block2]])

def loss(beta):
    return np.sum(res(beta)**2)

@np.vectorize
def loss_vectorized(beta1, beta2):
    beta = np.array([beta1, beta2])
    return loss(beta)

beta1, beta2 = np.meshgrid(np.linspace(-1.25,1.25,250),np.linspace(-1.25,1.25,250))

loss_ev = loss_vectorized(beta1, beta2)


levels = np.linspace(loss_ev.min(),loss_ev.max(),30)

plt.plot(figsize=(10,10))
contourplot = plt.contourf(beta1,beta2,loss_ev, levels=levels, cmap=cm.coolwarm)
cbar = plt.colorbar(contourplot, label=r"$\mathcal{L}$")
plt.plot(1,1, "ok",label="Minimum")
plt.plot(*beta0, "sk",label=r"$\beta_0$")




beta_list = [beta0]
def cb_beta(beta):
    global error_list, loss_list
    beta_list.append(beta.copy())

gauss_newton(res, beta0, jac, callback=cb_beta)

beta_array = np.array(beta_list).T

plt.plot(beta_array[0],beta_array[1], "x-k", label="gauss newton")

plt.savefig("rosenbrock_3d.png")
plt.legend()
plt.show()
