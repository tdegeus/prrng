import matplotlib.pyplot as plt
import numpy as np
import prrng

gen = prrng.pcg32()

fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for k, c in zip([1.0, 2.0, 4.0], ["r", "g", "b"]):

    x = np.logspace(-2, 1, 100)
    P = prrng.weibull_distribution(k).pdf(x)
    axes[0].plot(x, P, c=c)

    P, x = np.histogram(np.random.weibull(k, size=10000), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="dotted")

    P, x = np.histogram(gen.weibull([10000], k), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="--")

    # --

    x = np.linspace(0, 10, 1000)
    P = prrng.weibull_distribution(k).cdf(x)
    axes[1].plot(x, P, c=c)

    x = np.sort(np.random.weibull(k, size=10000))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="dotted")

    x = np.sort(gen.weibull([10000], k))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="--")

for ax in [axes[0]]:

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim([1e-3, 2e0])
    ax.set_xlim([1e-2, 2e1])

    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

for ax in [axes[1]]:

    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

plt.show()
