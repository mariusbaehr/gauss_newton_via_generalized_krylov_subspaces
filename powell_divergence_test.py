from gauss_newton import gauss_newton
import matplotlib.pyplot as plt
import numpy as np

beta_exact = np.array([0])

def res(beta,tau): return np.array([beta[0]+1,tau*beta[0]**2+beta[0]-1])
def jac(beta,tau): return np.array([[1], [2*tau*beta[0] +1]])
def loss(beta,tau): return np.sum(res(beta,tau)**2)
# beta_exact hier berechnen!!!!
def error_list(beta_list): return np.abs(beta_exact-beta_list)
beta0 = np.array([1.0])

def cb_beta(beta): 
    global beta_list
    beta_list.append(beta.copy())
       

beta_list = []
tau = -5
gauss_newton(res,beta0,jac, args = (tau,),callback=cb_beta)
beta_ag = beta_list


def no_step_length_control(*_):
    return 1

beta_list = []
tau = -5
gauss_newton(res,beta0,jac, args = (tau,), max_iter = 20, callback=cb_beta,step_length_control=no_step_length_control)
beta_without = beta_list

#tau = 5
#beta_list = []
#gauss_newton(res,beta0,jac,callback=cb_beta)
#gauss_

#
#
# Hier auch das andere Beispiel mit den zu kleinen Schrittweiten implementieren (Das erste wo die schrittweiten zu groß sind weglassen da es ja quasie schon das powell beispiel ist!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
#
# 

iter = 0
def to_small_steps(res , beta, jac_ev, descent_direction, *_):
    global iter
    step_length = descent_direction/np.linalg.norm(descent_direction)*2**-iter
    iter += 1
    return step_length


plt.figure(figsize=(10,10))
plt.yscale('log')
plt.plot(range(len(beta_ag)),error_list(beta_ag), "x-", label = "armijo goldstein")
#plt.plot(range(len(beta_without)),error_list(beta_without), "x-", label = "without")
plt.plot(range(len(beta_without)),1/np.arange(len(beta_without)), "--", label = r"$\Theta()$")

plt.legend()
plt.show()

