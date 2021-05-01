# prrng

[![CI](https://github.com/tdegeus/prrng/workflows/CI/badge.svg)](https://github.com/tdegeus/prrng/actions)
[![Doxygen -> gh-pages](https://github.com/tdegeus/prrng/workflows/gh-pages/badge.svg)](https://tdegeus.github.io/prrng)
[![readthedocs](https://readthedocs.org/projects/prrng/badge/?version=latest)](https://readthedocs.org/projects/prrng/badge/?version=latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/prrng.svg)](https://anaconda.org/conda-forge/prrng)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/python-prrng.svg)](https://anaconda.org/conda-forge/python-prrng)

Portable Reconstructible (Pseudo) Random Number Generator.

Documentation: https://tdegeus.github.io/prrng

## Credits

This library is a wrapper around [imneme/pcg-c-basic](https://github.com/imneme/pcg-c-basic), see also [pcg-random.org](http://www.pcg-random.org), and uses some features from [wjakob/pcg32](https://github.com/wjakob/pcg32).

## Overview

The idea is to provide a random number generator the can return nd-distributions of random numbers. 
In addition a bunch of random number generators can be collected in an nd-array, 
such that a composite array of random numbers is returned.

The key feature of this library is that is implements both the random number generator,
as well as the distributions, and provides a Python API.
In this way, given a certain seed, the same distribution can be generated from both C++ and Python,
on all platforms and for all compilers.
