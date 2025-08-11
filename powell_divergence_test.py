from gauss_newton import gauss_newton
import matplotlib.pyplot as plt
import numpy as np

tau = -5

def res(beta): return np.array([beta[0]+1,tau*beta[0]**2+beta[0]-1])
def jac(beta): return np.array([[1], [2*tau*beta[0] +1]])
def loss(beta): return np.sum(res(beta)**2)

beta0 = np.array([1.0])

beta_list = []
def cb_beta(beta): 
    global beta_list
    beta_list.append(beta.copy())
       

gauss_newton(res,beta0,jac,callback=cb_beta)
gn_armijo_goldstein = beta_list
loss_gn_armijo_goldstein = [loss(beta) for beta in gn_armijo_goldstein]


def no_step_length_control(*_):
    return 1

beta_list = []
gauss_newton(res,beta0,jac,max_iter = 20, callback=cb_beta,step_length_control=no_step_length_control)
gn_without = beta_list
loss_gn_without = [loss(beta) for beta in gn_without]
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
def to_small_steps(_, _, _, descent_direction, *_):
    global iter
    step_length = descent_direction/np.linalg.norm(descent_direction)*2**-iter
    iter += 1
    return step_length

plt.figure(figsize=(10,10))
plt.loglog(range(len(gn_armijo_goldstein)),loss_gn_armijo_goldstein, "x-", label="armijo goldstein")
plt.loglog(range(len(gn_without)),loss_gn_without, "x-", label = "without step lenght controll")
plt.legend()
plt.show()

