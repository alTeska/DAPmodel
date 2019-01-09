import numpy as np

from setuptools import Extension, setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize


with open("README.md", "r") as fh:
    long_description = fh.read()


extensions = [Extension('dap.dap_cython',
                       ['dap/dap_cython.pyx'],
                       include_dirs = [np.get_include()]),
              Extension('dap.dap_cython_be',
                       ['dap/dap_cython_be.pyx'],
                       include_dirs = [np.get_include()]),]


setup(
    name='dap',
    version='0.0.1',
    scripts=['dap_model'] ,
    author="Aleksandra Teska",
    author_email="aleksandra.teska@gmail.com",
    description="DAP model with multpile integration schemes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alTeska/DAPmodel",
    # packages=find_packages(),
    # packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    packages=['dap', 'dap.cell_fitting'],
    ext_modules = cythonize(extensions),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        ],
)
