import setuptools
from distutils.core import setup

setup(
    name='pyIID',
    version='',
    packages=['extra', 'pyiid', 'pyiid.sim', 'pyiid.calc', 'pyiid.tests',
              'pyiid.tests.test_master', 'pyiid.kernels', 'pyiid.testing',
              'pyiid.wrappers', 'pyiid.wrappers.mpi'],
    url='',
    license='',
    author='Christopher J. Wright',
    author_email='wright1@email.sc.edu',
    description='',
    requires=['numba', 'numpy', 'ase', 'nose', 'mpi4py', 'simdb', 'matplotlib',
              'matplotlib']
)