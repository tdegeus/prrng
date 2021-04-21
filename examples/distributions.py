import prrng
import numpy as np
import matplotlib.pyplot as plt

gen = prrng.pcg32()

fig, ax = plt.subplots()

for k, c in zip([1.0, 2.0, 4.0], ['r', 'g', 'b']):

    P, x = np.histogram(np.random.weibull(k, size=10000), bins=100)
    x = 0.5 * (x[1:] + x[:-1])
    ax.plot(x, P, c=c)

    P, x = np.histogram(gen.weibull([10000], k), bins=100)
    x = 0.5 * (x[1:] + x[:-1])
    ax.plot(x, P, c=c, ls='--')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel('x')
ax.set_ylabel('P(x)')

plt.show()

