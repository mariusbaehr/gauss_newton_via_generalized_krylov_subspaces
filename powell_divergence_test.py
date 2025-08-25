from gauss_newton import gauss_newton
from armijo_goldstein import armijo_goldstein
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize


def res(beta,tau): return np.array([beta[0]+1,tau*beta[0]**2+beta[0]-1])
def jac(beta,tau): return np.array([[1], [2*tau*beta[0] +1]])
def loss(beta,tau): return np.sum(res(beta,tau)**2)
# beta_exact hier berechnen!!!!
beta0 = np.array([1.0])

def cb_beta(beta): 
    global beta_list
    beta_list.append(beta.copy())

max_iter=20
      
# No step length control, i.e. step length = 1
def no_step_length_control(*_):
    return 1

# To big but feasible
# Idea: beta + step_length*descent_direction = beta_new = (-1+2**-iter)*beta
# so (-2+2**-iter)*beta/descent_direction = step_length

# Bug: Dosnt work on this function, either modify, take beta**2 or leave it out

iter = 1
def too_big_steps(res , beta, jac_ev, args, descent_direction, *_):
    global iter
    step_length =(-2+2**(-iter))*beta[0]/descent_direction[0] # Notice that descent_direction is an scalar array
    iter += 1
    return step_length


# To small, but feasible
iter = 2
def too_small_steps(res , beta, jac_ev, args, descent_direction, *_):
    global iter
    step_length = -1/descent_direction[0]*2**-iter
    iter += 1
    return step_length

#TODO: Test if newton has quadratic convergence
#beta_list = []
#def cb_newton(xk):
#    global beta_list
#    beta_list.append(xk)

#scipy.optimize.
    


# Plot 
fig, ax = plt.subplots(1,2,figsize=(20,10))


beta_list = []
tau = -5
beta_exact =np.array([0]) # scipy.optimize.minimize(loss,beta0,args=(tau),bounds=[(0.1,1)]).x # In case of tau>1, the method converges to the minimum nearest to beta0, 
# makes no sense doing it this way
def error_list(beta_list): return np.abs(beta_exact-beta_list)
beta_min=0
beta_max=0
#step_length_controls=[armijo_goldstein,too_big_steps,too_small_steps,no_step_length_control]
#colors = ["tab:green",  "tab:purple","tab:orange", "tab:red"]

step_length_controls=[armijo_goldstein,too_small_steps ,no_step_length_control]
colors = ["tab:green", "tab:orange", "tab:red"]
linestyles = ["-","-", "-."]
for step_length_control, color, linestyle in zip(step_length_controls,colors,linestyles):

    beta_list = [beta0]
    gauss_newton(res,beta0,jac, args = (tau,), max_iter = max_iter, callback=cb_beta, step_length_control=step_length_control)
    ax[0].plot(beta_list, [loss(np.array([beta]),tau) for beta in beta_list], "x", color=color)

    ax[1].semilogy(range(len(beta_list)),error_list(beta_list), "x", linestyle=linestyle, label = step_length_control.__name__, color=color)
    beta_min = min(np.min(beta_list),beta_min)
    beta_max = max(np.max(beta_list),beta_max)

beta_span = np.linspace(beta_min,beta_max,100)
ax[0].plot(beta_span, [loss(np.array([beta]),tau) for beta in beta_span], color="tab:blue")

ax[1].semilogy(range(len(beta_list)),10.0**(-1*np.arange(len(beta_list))), "--", label = r"$\Theta(e_k^1)$")

#for k in range(max_iter+1):
#    ax[0].text(beta_list[k],loss(np.array([beta_list[k]]),tau),str(k),ha="center", va="bottom", color="tab:red", fontsize=10)


ax[1].legend()

plt.savefig('powell_divergence.png')
plt.show()




