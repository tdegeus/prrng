import prrng
import numpy as np
import matplotlib.pyplot as plt

gen = prrng.pcg32()

fig, ax = plt.subplots()

for k, c in zip([1.0, 2.0, 4.0], ['r', 'g', 'b']):

    x = np.logspace(-2, 1, 100)
    P = prrng.gamma_distribution(k).pdf(x)
    ax.plot(x, P, c=c)

    P, x = np.histogram(np.random.gamma(k, size=10000), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    ax.plot(x, P, c=c, ls='dotted')

    P, x = np.histogram(gen.gamma([10000], k), bins=100, density=True)
    x = 0.5 * (x[1:] + x[:-1])
    ax.plot(x, P, c=c, ls='--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim([1e-3, 2e0])
ax.set_xlim([1e-2, 2e1])

ax.set_xlabel('x')
ax.set_ylabel('P(x)')

plt.show()

