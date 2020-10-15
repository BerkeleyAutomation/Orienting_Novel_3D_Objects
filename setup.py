"""
Setup of core python codebase
Author: Ashwin Balakrishna, Chris Correa
"""
import os
from setuptools import setup
from Cython.Build import cythonize
import numpy
from setuptools.extension import Extension

requirements = [
    'numpy',
    'matplotlib',
]
setup_requirements = [
    'Cython',
    'numpy'
]
libraries = []
if os.name == 'posix':
    libraries.append('m')
extra_compile_args = ["-O3", "-ffast-math", "-march=native"]
extra_link_args = []

extensions = [
]

setup(
    name='unsupervised_rbt',
    version = '0.1.0',
    description = 'Unsupervised Representation Learning by Predicting Rigid Body Transformations in Depth Images',
    author = 'Ashwin Balakrishna, Chris Correa',
    author_email = 'ashwin_balakrishna@berkeley.edu, chris.correa@berkeley.edu',
    url = 'https://github.com/BerkeleyAutomation/unsupervised_rbt',
    package_dir = {'': '.'},
    packages = ['unsupervised_rbt'],
    install_requires = requirements,
    setup_requires=setup_requirements,
    ext_modules=cythonize(extensions, gdb_debug=False)

)
