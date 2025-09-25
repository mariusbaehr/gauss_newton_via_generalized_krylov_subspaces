from gauss_newton import gauss_newton
from armijo_goldstein import armijo_goldstein
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def res(x, tau):
    return np.array([x[0] + 1, tau * x[0] ** 2 + x[0] - 1])


def jac(x, tau):
    return np.array([[1], [2 * tau * x[0] + 1]])


def loss(x, tau):
    return np.sum(res(x, tau) ** 2)


x0 = np.array([1.0])

max_iter = 19


def cb_x(x):
    global x_list
    x_list.append(x.copy())


# No step length control, i.e. step length = 1
def no_step_length_control(res, x, res_ev, jac_ev, args, descent_direction, *_):
    return 1, res(x + descent_direction, *args), 1


# To small, but feasible
iter = 2


def too_small_steps(res, x, res_ev, jac_ev, args, descent_direction, *_):
    global iter
    step_length = -1 / descent_direction[0] * 2**-iter
    iter += 1

    # Check if step length would by feasible according Armijo rule
    prev_loss = np.sum(res(x, *args) ** 2)
    jac_dot_descent = np.sum((jac_ev @ descent_direction) ** 2) / 2
    res_ev = res(x + step_length * descent_direction, *args)
    current_loss = np.sum(res_ev**2)
    if prev_loss - current_loss < step_length * jac_dot_descent:
        print("too_small_steps: step was rejected by the Armijo rule")

    return step_length, res(x + step_length * descent_direction, *args), 1


# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


x_list = []
tau = -5
x_exact = np.array([0])
ax[0].plot(
    x_exact, loss(x_exact, tau), "ks", label=r"$x^\ast$", zorder=-1, markersize=7
)


def error_list(x_list):
    return np.abs(x_exact - x_list)


x_min = 0
x_max = 0

step_length_controls = [armijo_goldstein, too_small_steps, no_step_length_control]
colors = ["tab:green", "tab:blue", "tab:orange"]
linestyles = ["-", "-", "-."]
markers = ["o", "+", "x"]
for step_length_control, color, linestyle, marker in zip(
    step_length_controls, colors, linestyles, markers
):
    x_list = [x0]
    gauss_newton(
        res,
        x0,
        jac,
        args=(tau,),
        max_iter=max_iter,
        callback=cb_x,
        step_length_control=step_length_control,
    )
    ax[0].scatter(
        x_list, [loss(np.array([x]), tau) for x in x_list], marker=marker, color=color
    )

    ax[1].semilogy(
        range(len(x_list)),
        error_list(x_list),
        marker=marker,
        linestyle=linestyle,
        label=step_length_control.__name__,
        color=color,
    )
    x_min = min(np.min(x_list), x_min)
    x_max = max(np.max(x_list), x_max)

x_span = np.linspace(x_min, x_max, 100)
ax[0].plot(
    x_span,
    [loss(np.array([x]), tau) for x in x_span],
    color="black",
    zorder=-1,
    label=r"$\mathcal{L}$",
)

for k in range(10):
    if k == 3 or k == 2:
        continue
    ax[0].text(
        x_list[k],
        loss(np.array([x_list[k]]), tau),
        str(k),
        ha="left",
        va="bottom",
        color="tab:orange",
        fontsize=10,
    )
ax[0].text(
    x_list[3],
    loss(np.array([x_list[3]]), tau),
    str(3),
    ha="right",
    va="bottom",
    color="tab:orange",
    fontsize=10,
)
ax[0].text(
    x_list[2],
    loss(np.array([x_list[2]]), tau),
    str(2),
    ha="right",
    va="bottom",
    color="tab:orange",
    fontsize=10,
)

ax[0].set_xlabel("$x$")
ax[0].set_ylabel(r"$\mathcal{L}$")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Fehler")
ax[0].legend()
ax[1].legend()
ax[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.savefig("powell_divergence.png", bbox_inches="tight")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

x_list = []
tau = 5
x_exact = np.array([np.sqrt(16 * tau - 7) / (4 * tau) - 3 / (4 * tau)])
ax[0].plot(
    x_exact, loss(x_exact, tau), "ks", label=r"$x^\ast$", zorder=-1, markersize=7
)


def error_list(x_list):
    return np.abs(x_exact - x_list)


x_min = 0
x_max = 0

step_length_controls = [armijo_goldstein, no_step_length_control]
colors = ["tab:green", "tab:orange"]
linestyles = ["-", "--"]
markers = ["o", "x"]
for step_length_control, color, linestyle, marker in zip(
    step_length_controls, colors, linestyles, markers
):
    x_list = [x0]
    gauss_newton(
        res,
        x0,
        jac,
        args=(tau,),
        max_iter=max_iter,
        callback=cb_x,
        step_length_control=step_length_control,
    )
    ax[0].scatter(
        x_list, [loss(np.array([x]), tau) for x in x_list], marker=marker, color=color
    )

    ax[1].semilogy(
        range(len(x_list)),
        error_list(x_list),
        marker=marker,
        linestyle=linestyle,
        label=step_length_control.__name__,
        color=color,
    )
    x_min = min(np.min(x_list), x_min)
    x_max = max(np.max(x_list), x_max)

x_span = np.linspace(x_min, x_max, 100)
ax[0].plot(
    x_span,
    [loss(np.array([x]), tau) for x in x_span],
    color="black",
    zorder=-1,
    label=r"$\mathcal{L}$",
)

ax[0].set_xlabel("$x$")
ax[0].set_ylabel(r"$\mathcal{L}$")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Fehler")
ax[0].legend()
ax[1].legend()
plt.savefig("powell_convergence.png", bbox_inches="tight")
plt.show()
