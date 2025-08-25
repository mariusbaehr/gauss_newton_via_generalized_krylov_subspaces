import numpy as np
import scipy.sparse 
from benchmark import benchmark


P = 100 # N=2P-2, again not a classical regression model, however 
def res(beta):
    block1 = 10*(beta[1:]-beta[:-1]**2)
    block2 = 1-beta[:-1]
    return np.concatenate([block1,block2])

beta0 = np.zeros(P)
beta0[0]=1 #Because zero is not allowed in gauss_newton_krylov

def jac(beta):
    block1 = 10*scipy.sparse.eye(P-1,P,k=1) - 20*scipy.sparse.diags(beta[:-1],shape=(P-1,P))
    block2 = -scipy.sparse.eye(P-1,P,k=0)
    return scipy.sparse.block_array([[block1],[block2]])

beta_exact = np.ones(P)

def error(beta):
    return np.linalg.norm(beta-beta_exact)

benchmark(res, beta0, jac, error, {'max_iter': 500 })

#TODO: Plot the step length, I guess they are set to 1 when the error goes down

