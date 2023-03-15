import matplotlib.pyplot as plt
import numpy as np
import prrng

gen = prrng.pcg32()

fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for (mu, sigma), c in zip([(1.0, 1.0), (1.0, 0.5), (-2.0, 1.0), (-2.0, 2.0)], ["r", "g", "b", "c"]):
    x = np.linspace(-10, 10, 1000)
    P = prrng.normal_distribution(mu, sigma).pdf(x)
    axes[0].plot(x, P, c=c)

    P, x = np.histogram(np.random.normal(mu, sigma, size=10000), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="dotted")

    P, x = np.histogram(gen.normal([10000], mu, sigma), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    axes[0].plot(x, P, c=c, ls="--")

    # --

    x = np.linspace(-10, 10, 1000)
    P = prrng.normal_distribution(mu, sigma).cdf(x)
    axes[1].plot(x, P, c=c)

    x = np.sort(np.random.normal(mu, sigma, size=10000))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="dotted")

    x = np.sort(gen.normal([10000], mu, sigma))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, P, c=c, ls="--")

for ax in [axes[0]]:
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

for ax in [axes[1]]:
    ax.set_xlabel("x")
    ax.set_ylabel("P(x)")

plt.show()
