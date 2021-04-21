
from setuptools import setup, Extension

import re
import os
import pybind11
import pyxtensor
from setuptools_scm import get_version

version = get_version()

include_dirs = [
    os.path.abspath('include/'),
    pyxtensor.find_pyxtensor(),
    pyxtensor.find_pybind11(),
    pyxtensor.find_xtensor(),
    pyxtensor.find_xtl()]

build = pyxtensor.BuildExt

build.c_opts['unix'] += ['-DPRRNG_VERSION="{0:s}"'.format(version)]
build.c_opts['msvc'] += ['/DPRRNG_VERSION="{0:s}"'.format(version)]

ext_modules = [Extension(
    'prrng',
    ['python/main.cpp'],
    include_dirs = include_dirs,
    language='c++')]

setup(
    name = 'prrng',
    description = 'Portable Reconstructible Random Number Generator',
    long_description = 'Portable Reconstructible Random Number Generator',
    keywords = 'random',
    version = version,
    license = 'MIT',
    author = 'Tom de Geus',
    author_email = 'tom@geus.me',
    url = 'https://github.com/tdegeus/prrng',
    ext_modules = ext_modules,
    setup_requires = ['pybind11', 'pyxtensor'],
    cmdclass = {'build_ext': build},
    zip_safe = False)
