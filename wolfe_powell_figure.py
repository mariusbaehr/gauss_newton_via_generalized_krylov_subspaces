import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

sigma = 0.1
rho = 0.35

loss = Polynomial.fit([0,1,2,4,5],[3,1,2,0,3],4, domain = [0,5])
slope_loss = loss.deriv(1)
def sufficient_decrease(alpha): return loss(0)+slope_loss(0)*sigma*alpha


alphas = np.linspace(0,5,200)

plt.figure(figsize=(12,6))
plt.xlim(-0.5, 5.5)
plt.ylim(-1.2, 3.5)
plt.annotate("", xytext=(5.2, 0), xy=(-0.2, 0), arrowprops=dict(arrowstyle="<-"))
plt.annotate("", xytext=(0,3.2), xy=(0, -0.2), arrowprops=dict(arrowstyle="<-"))
plt.axis("off")

loss_ev=loss(alphas)
sufficient_decrease_ev = sufficient_decrease(alphas)
plt.plot(alphas, loss(alphas), color="black", label=r"$\mathcal{L}(x+\alpha d)$")
plt.plot(alphas,sufficient_decrease_ev, label=r"$t(\alpha)$")

sufficient_decrease_equality = (loss - loss(0) -slope_loss(0)*sigma*Polynomial.identity().convert(domain=[0,5])).roots()
print(sufficient_decrease_equality)
plt.plot(sufficient_decrease_equality[:2], [-1 ,-1], lw=5, color="tab:blue", label="Hinreichenden Abstiegsbedingung")
plt.plot(sufficient_decrease_equality[2:], [-1 ,-1], lw=5, color="tab:blue")
plt.text(sufficient_decrease_equality[1], loss(sufficient_decrease_equality[1]) + 0.1, r"$\alpha'$")
plt.plot(sufficient_decrease_equality[1], loss(sufficient_decrease_equality[1]), "ko")

curvature_condition_equality = (slope_loss - slope_loss(0)*rho).roots()
print(curvature_condition_equality)
point = curvature_condition_equality[0].real
def curvature_condition(alpha): return loss(point)+slope_loss(0)*rho*(alpha-point)
plt.plot(alphas[:48],curvature_condition(alphas)[:48], label=r"$\mathcal{L}(\alpha''')+\rho\alpha\mathcal{L}'(0)$")
plt.plot([point, 5], [-0.5 ,-0.5], lw=5, label="Krümmungsbedingnung", color="tab:orange")
plt.plot(point,loss(point), "ko")
plt.text(point,loss(point)+0.1, r"$\alpha'''$")

alpha_prime2 = (slope_loss - slope_loss(0)*sigma).roots()
print(alpha_prime2)
point = alpha_prime2[0].real
def alpha_prime2_line(alpha): return loss(point)+slope_loss(0)*sigma*(alpha-point)
plt.plot(alphas[:114],alpha_prime2_line(alphas)[:114], linestyle="--", color="tab:gray", label=r"$\mathcal{L}(\alpha'')+\rho\alpha\mathcal{L}'(0)$")
plt.plot(point,loss(point), "o", color="tab:gray")
plt.text(point,loss(point)+0.1, r"$\alpha''$", color ="tab:gray")

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.75)
plt.savefig('wolfe_powell.png',bbox_inches='tight')
plt.show()

