import GooseMPL as gplt
import matplotlib.pyplot as plt
import numpy as np
import prrng

plt.style.use(["goose", "goose-latex", "goose-autolayout"])

gen = prrng.pcg32()

fig, axes = plt.subplots(figsize=(16, 6), ncols=2)

for k, c in zip([1.0, 2.0, 4.0], ["r", "g", "b"]):

    r = np.random.pareto(k, size=10000) + 1
    bin_edges = gplt.histogram_bin_edges(r, bins=80, mode="log")
    P, x = gplt.histogram(r, bins=bin_edges, density=True, return_edges=False)
    axes[0].plot(x, P, c=c, ls="dotted")

    r = gen.pareto([10000], k)
    bin_edges = gplt.histogram_bin_edges(r, bins=80, mode="log")
    P, x = gplt.histogram(r, bins=bin_edges, density=True, return_edges=False)
    axes[0].plot(x, P, c=c)

    x = np.logspace(0, 3, 100)
    P = prrng.pareto_distribution(k).pdf(x)
    axes[0].plot(x, P, c=c, lw=1)

    # --

    x = np.sort(np.random.pareto(k, size=10000) + 1)
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, 1 - P, c=c, ls="dotted")

    x = np.sort(gen.pareto([10000], k))
    P = np.linspace(0, 1, x.size)
    axes[1].plot(x, 1 - P, c=c, ls="--")

    x = np.logspace(0, 3, 100)
    P = prrng.pareto_distribution(k).cdf(x)
    axes[1].plot(x, 1 - P, c=c, lw=1)

for ax in [axes[0]]:

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$P(x)$")

for ax in [axes[1]]:

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\Phi(x)$")

plt.show()
