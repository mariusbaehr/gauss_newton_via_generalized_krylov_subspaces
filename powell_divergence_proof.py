import sympy as sp

x, tau = sp.symbols("x tau")

r = sp.Matrix([[x + 1], [tau * x**2 + x - 1]])

loss = 1/2*(r[0, 0] ** 2 + r[1, 0] ** 2)
critical_points = sp.roots(sp.diff(loss, x), x)
print(
    f"critical points of loss: {critical_points}"
)  # As one can easily see the first two roots are reel iff tau >=7/16
for critical_point in critical_points:
    print(sp.simplify(sp.diff(loss, x, x).subs(x, critical_point)))
# Furthermore 0 is the only local and there for global minimizer of loss, iff tau<1, as one can see from the last term
# If however tau>1 then 0 is a local maximum of loss, there for the following calculations become irrelevant for tau>1

jac = sp.diff(r, x)
jac_pseudoinverse = (jac.T @ jac) ** -1 @ jac.T  # Rank is always 1

x_next = x - (jac_pseudoinverse @ r)[0]
taylor = sp.series(x_next, x, 0, 3)  # 3rd degree Taylor polynomial at 0

print(f"3rd degree Taylor polynomial at 0 for next iteration: {taylor}")
