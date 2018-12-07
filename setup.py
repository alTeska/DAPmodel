import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages


extensions = [Extension('DAPmodel.DAPbaseC',
                        ['DAPmodel/DAPbaseC.pyx'],
                        include_dirs = [np.get_include()])]


setup(
    name='DAPmodel',
    version='0.0.1',
    install_requires=['numpy', 'scipy', 'delfi'],
    ext_modules = cythonize(extensions)
)
