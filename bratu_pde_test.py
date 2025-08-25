import numpy as np
import scipy.sparse 
from benchmark import benchmark

n =  25 # Notice N=P=(n-1)*(n-1)
      # Not actually a regression problem with gaussian noise, because N=P. However at least the linear case a2=0 might be interesting for checking if the generalized krylov subspaces are correct implemented.
a1 = 5
a2 = 0

a=0
b=1
h=1#(b-a)/n #Note the scaling was omitted in "An effective implementation of Gauss Newton via generalized Krylov subspaces"

def u(x1,x2): return np.exp(-10*(x1**2+x2**2))

A_h1 = h**-2*scipy.sparse.diags_array((-np.ones(n-2),2*np.ones(n-1),-np.ones(n-2)),offsets=(-1,0,1)) # shape (n-1,n-1)
A_h2 = scipy.sparse.kron(A_h1,np.eye(n-1))+scipy.sparse.kron(np.eye(n-1),A_h1)                       # shape ((n-1)**2,(n-1)**2)
d_x1 = h**-1*scipy.sparse.kron(scipy.sparse.diags_array((-np.ones(n-1),np.ones(n-2)),offsets=(0,1)),np.eye(n-1)) #shape ((n-1)**2,(n-1)**2)

x1, x2 = np.meshgrid(np.linspace(a,b,n+1)[1:-1],np.linspace(a,b,n+1)[1:-1])

x_true = u(x1,x2).flatten()
y_true = A_h2@x_true+a1*d_x1@x_true+a2*np.exp(x_true)

def res(beta): return y_true-A_h2@beta-a1*d_x1@beta-a2*np.exp(beta)
beta0 = np.zeros_like(x_true)
beta0[1] = 1
#beta0= A_h2.T@y_true+a1*d_x1.T@y_true
#beta0 /= np.linalg.norm(beta0)
def jac(beta): return -A_h2-a1*d_x1-a2*scipy.sparse.diags(np.exp(beta))
def error(beta): return np.linalg.norm(x_true-beta)



def a_Ta(beta):
    return (A_h2+a1*d_x1).T@((A_h2+a1*d_x1)@beta)
A_TA = scipy.sparse.linalg.LinearOperator(((n-1)**2,(n-1)**2),matvec=a_Ta)


def ref_cg(callback):
    cg, _ = scipy.sparse.linalg.cg(A_TA, A_h2.T@y_true+a1*d_x1.T@y_true, beta0, callback=callback)
    return cg

benchmark(res,beta0, jac, error, {'max_iter':300}, additional_methods=[ref_cg])

#print(f"condition number of A.T@A {np.linalg.cond( ((A_h2+a1*d_x1).T@(A_h2+a1*d_x1)).todense())}")
#print(f"norm of A.T@A {np.linalg.norm(((A_h2+a1*d_x1).T@(A_h2+a1*d_x1)).todense(),2)}")

#import matplotlib.pyplot as plt
#spec = np.linalg.eigvals(((A_h2+a1*d_x1).T@((A_h2+a1*d_x1))).todense())
#plt.figure()
#plt.scatter(np.real(spec),np.imag(spec))
#plt.scatter(0,0,label="Null")
#plt.show()


