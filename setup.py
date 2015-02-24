from distutils.core import setup

setup(
    name='pyIID',
    version='1',
    packages=['extra', 'pyiid', 'pyiid.sim', 'pyiid.calc', 'pyiid.test_kernels',
              'pyiid.old_hmc', 'pyiid.scripts', 'pyiid.scripts.benchmarking',
              'pyiid.test_wrappers'],
    url='',
    license='',
    author='christopher',
    author_email='wright1@email.sc.edu',
    description='Hamiltonion Monte Carlo based Diffraction Simulation',
    requires=['numpy', 'python-ase', 'numba', 'nose', 'matplotlib']
)
