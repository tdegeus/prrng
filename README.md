# prrng

[![CI](https://github.com/tdegeus/prrng/workflows/CI/badge.svg)](https://github.com/tdegeus/prrng/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/prrng/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/prrng)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/prrng.svg)](https://anaconda.org/conda-forge/prrng)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-prrng.svg)](https://anaconda.org/conda-forge/python-prrng)

Portable Reconstructible (Pseudo) Random Number Generator.

Documentation: https://tdegeus.github.io/prrng

## Overview

### Credits

This library is a wrapper around [imneme/pcg-c-basic](https://github.com/imneme/pcg-c-basic), see also [pcg-random.org](http://www.pcg-random.org), and uses some features from [wjakob/pcg32](https://github.com/wjakob/pcg32).

### Basic example

The idea is to provide a random number generator the can return nd-distributions of random numbers. Example:

```cpp
#include <ctime>
#include <xtensor/xtensor.h>
#include <prrng.h>

int main()
{
    auto seed = std::time(0);
    prrng::pcg32 generator(seed);
    auto a = generator.random({10, 40});
    return 0;
}
```

### pcg32

One of the hallmark features of the pcg32 generator is that the state of the random generator (the position in random sequence set by the seed) can be saved and restored at any point

```cpp
#include <xtensor/xtensor.h>
#include <prrng.h>

int main()
{
    prrng::pcg32 generator();
    auto state = generator.state();
    auto a = generator.random({10, 40});
    generator.restore(state);
    auto b = generator.random({10, 40});
    assert(xt::allclose(a, b));
    return 0;
}
```

In addition one can advance and reverse in the random sequence, and compute the number of random numbers between two states.

**Important:** A very important and hallmark features of pcg32 is that, internally, types of fixed bit size are used. Notably the state is (re)stored as `uint64_t`. This makes that restoring can be
uniquely done on any system and any compiler, on any platform (as long as you save the `uint64_t` properly, naturally).
As a convenience the output can be recast by specifying a template parameter, while static assertions shield you from losing data. For example, `auto state = generator.state<size_t>();` would be allowed, but `auto state = generator.state<int>();` would not.

### Python API

A Python API is provided allowing one to obtain the same random sequence from C++ and Python when the same seed is used.
In fact, a generator can be stored and restored in any of the two languages.
Example:

```python
import prrng

generator = prrng.pcg32()
state = generator.state()
a = generator.random([10, 40])
generator.restore(state)
b = generator.random([10, 40])
assert np.allclose(a, b)
```

### Array of tensors

In addition a bunch of random number generators can be collected in an nd-array,
such that a composite array of random numbers is returned.
Example:

```cpp
#include <xtensor/xtensor.h>
#include <prrng.h>

int main()
{
    xt::xtensor<uint64_t, 2> seed = {{0, 1, 2}, {3, 4, 5}};
    prrng::pcg32_array generator(seed);
    auto state = generator.state();
    auto a = generator.random({4, 5}); // shape {2, 3, 4, 5}
    generator.restore(state);
    auto b = generator.random({4, 5});
    assert(xt::allclose(a, b));
    return 0;
}
```

### Random distributions

Each random generator can return a random sequence according to a certain distribution.
The most basic behaviour is to just convert a random integer to a random double `[0, 1]`,
as was already done in the examples using `generator.random<...>(...)`.
Also this feature is included in the Python API, allowing to get a reproducible distribution.

### More information

*   The documentation of the code.
*   The code itself.
*   The unit tests, under [tests](./tests).
*   The examples, under [examples](./examples).

## Change-log

### v0.6.0

*   [Python] setup.py: support cross-compilation, allowing customization
*   [CMake] allowing simd
*   [CMake] Avoiding setuptools_scm dependency if SETUPTOOLS_SCM_PRETEND_VERSION is defined
*   [CI] Minor updates
*   [docs] Updating doxystyle
*   [docs] Building docs on release
*   Using new operators xtensor
*   Minor style update
*   Fixing weibull_distribution::cdf. Plotting CDF
*   [CMake] Minor updates
