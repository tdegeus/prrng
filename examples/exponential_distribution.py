import matplotlib.pyplot as plt
import numpy as np
import prrng

gen = prrng.pcg32()

fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for scale, c in zip([1.0, 2.0, 3.0, 4.0], ["r", "g", "b", "c"]):

    x = np.linspace(0, 50, 1000)
    P = prrng.exponential_distribution(scale).pdf(x)
    axes[0].plot(x, P, c=c)

    P, x = np.histogram(np.random.exponential(scale, size=10000), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="dotted")

    P, x = np.histogram(gen.exponential([10000], scale), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="--")

    # --

    x = np.linspace(0, 50, 1000)
    P = prrng.exponential_distribution(scale).cdf(x)
    axes[1].plot(x, P, c=c)

    x = np.sort(np.random.exponential(scale, size=10000))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="dotted")

    x = np.sort(gen.exponential([10000], scale))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="--")

for ax in [axes[0]]:

    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

for ax in [axes[1]]:

    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

plt.show()
