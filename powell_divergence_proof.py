import sympy as sp

beta, tau = sp.symbols('beta tau')

r = sp.Matrix([[beta+1],
               [tau*beta**2+beta-1]])
jac = sp.diff(r,beta)
jac_pseudoinverse = (jac.T@jac)**-1@jac.T

beta_next = beta - (jac_pseudoinverse@r)[0]
taylor = sp.series(beta_next,beta,0,2)

print(taylor)
