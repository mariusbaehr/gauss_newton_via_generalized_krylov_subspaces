import matplotlib.pyplot as plt
import numpy as np
import scipy

###########################################
# Für realismus habe ich jetzt erstmal das genommen, evtl. nacher noch ändern!!!!
tau = -5
def res(beta): return np.array([beta[0]+1,tau*beta[0]**2+beta[0]-1])
def jac(beta): return np.array([1, 2*tau*beta[0] +1])
def loss(beta): return np.sum(res(beta)**2)
beta = np.array([0.4])
##############################################

descent_direction, *_ = scipy.linalg.lstsq(-1*np.atleast_2d(jac(beta)).T, res(beta))

span_step_length = np.linspace(0,1.5,64)

@np.vectorize
def loss_along_descent(step_length): 
    return loss(beta+step_length*descent_direction)

#def loss_along_descent(span_step_length): 
#    return [loss(beta+step_length*descent_direction) for step_length in span_step_length]


plt.figure(figsize=(10,10))
plt.plot(span_step_length,loss_along_descent(span_step_length),label=r"$\alpha \mapsto \mathcal{L}(\beta+\alpha d)$")
#plt.title("Armijo Bedinung")
plt.legend()
plt.show()


