from distutils.core import setup

setup(
    name='pyIID',
    version='1',
    packages=[
        # 'extra',
        'pyiid',
        'pyiid.sim',
        'pyiid.calc',
        'pyiid.tests',
        # 'pyiid.old_hmc',
        'pyiid.scripts',
        'pyiid.scripts.benchmarking',
        'pyiid.tests.test_kernels',
        'pyiid.tests.test_wrappers'
    ],
    url='',
    license='',
    author='christopher',
    author_email='wright1@email.sc.edu',
    description='Hamiltonion Monte Carlo based Diffraction Simulation',
    requires=['numpy', 'ase', 'numba', 'nose', 'matplotlib', 'six']
)
