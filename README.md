# prrng

[![CI](https://github.com/tdegeus/prrng/workflows/CI/badge.svg)](https://github.com/tdegeus/prrng/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/prrng/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/prrng)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/prrng.svg)](https://anaconda.org/conda-forge/prrng)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-prrng.svg)](https://anaconda.org/conda-forge/python-prrng)

Portable Reconstructible (Pseudo) Random Number Generator.

Documentation: https://tdegeus.github.io/prrng

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

## Implementation

### C++ and Python

The code is a C++ header-only library,
but a Python module is also provided.
The interfaces are identical except:

+   All *xtensor* objects (`xt::xtensor<...>`) are *NumPy* arrays in Python.
+   All `::` in C++ are `.` in Python.

### Installation

#### C++ headers

##### Using conda

```bash
conda install -c conda-forge prrng
```

##### From source

```bash
# Download prrng
git checkout https://github.com/tdegeus/prrng.git
cd prrng

# Install headers, CMake and pkg-config support
cmake .
make install
```

#### Python module

##### Using conda

```bash
conda install -c conda-forge python-prrng
```

Note that *xsimd* and hardware optimisation are **not enabled**.
To enable them you have to compile on your system, as is discussed next.

##### From source

>   You need *xtensor*, *xtensor-python* and optionally *xsimd* as prerequisites.
>   In addition *scikit-build* is needed to control the build from Python.
>   The easiest is to use *conda* to get the prerequisites:
>
>   ```bash
>   conda install -c conda-forge xtensor-python
>   conda install -c conda-forge xsimd
>   conda install -c conda-forge scikit-build
>   ```
>
>   If you then compile and install with the same environment
>   you should be good to go.
>   Otherwise, a bit of manual labour might be needed to
>   treat the dependencies.

```bash
# Download prrng
git checkout https://github.com/tdegeus/prrng.git
cd prrng

# Compile and install the Python module
# (-vv can be omitted as is controls just the verbosity)
python setup.py install --build-type Release -vv

# OR, Compile and install the Python module with hardware optimisation
# (with scikit-build CMake options can just be added as command-line arguments)
python setup.py install --build-type Release -DUSE_SIMDD=1 -vv
```

### Compiling user-code

#### Using CMake

##### Example

Using *prrng* your `CMakeLists.txt` can be as follows

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(prrng REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE prrng)
```

##### Targets

The following targets are available:

*   `prrng`
    Includes *prrng* and the *xtensor* dependency.

*   `prrng::assert`
    Enables assertions by defining `QPOT_ENABLE_ASSERT`.

*   `prrng::debug`
    Enables all assertions by defining
    `QPOT_ENABLE_ASSERT` and `XTENSOR_ENABLE_ASSERT`.

*   `prrng::compiler_warings`
    Enables compiler warnings (generic).

##### Optimisation

It is advised to think about compiler optimisation and enabling *xsimd*.
Using *CMake* this can be done using the `xtensor::optimize` and `xtensor::use_xsimd` targets.
The above example then becomes:

```cmake
cmake_minimum_required(VERSION 3.1)
project(example)
find_package(prrng REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE
    prrng
    xtensor::optimize
    xtensor::use_xsimd)
```

See the [documentation of xtensor](https://xtensor.readthedocs.io/en/latest/) concerning optimisation.

#### By hand

Presuming that the compiler is `c++`, compile using:

```
c++ -I/path/to/prrng/include ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimisation,
enabling *xsimd*, ...

#### Using pkg-config

Presuming that the compiler is `c++`, compile using:

```
c++ `pkg-config --cflags prrng` ...
```

Note that you have to take care of the *xtensor* dependency, the C++ version, optimization,
enabling *xsimd*, ...

## Change-log

### v1.2.0

#### New features

*   Allow scalar return type where possible
*   Adding `randint`
*   Adding `delta` distribution (just to provide a quick API)
*   [docs] Using default doxygen theme

#### Internal changes

*   Making `normal_distribution::quantile` more robust
*   Omitting unneeded `is_xtensor`
*   [tests] Updating catch2 v3

### v0.6.1

*   Switching to scikit-build, clean-up of CMake (#24)

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
