"""
Setup of core python codebase
Author: Ashwin Balakrishna, Chris Correa
"""
from setuptools import setup

requirements = [
    'numpy',
    'matplotlib',
]

setup(
    name='unsupervised_rbt',
    version = '0.1.0',
    description = 'Unsupervised Representation Learning by Predicting Rigid Body Transformations in Depth Images',
    author = 'Ashwin Balakrishna, Chris Correa',
    author_email = '???, chris.correa@berkeley.edu',
    url = 'https://github.com/BerkeleyAutomation/unsupervised_rbt',
    package_dir = {'': '.'},
    packages = ['unsupervised_rbt'],
    install_requires = requirements,
)
