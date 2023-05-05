import time

import prrng

tic = time.time()
gen = prrng.pcg32()
for i in range(1000):
    a = gen.random([1000000])
print(time.time() - tic)
