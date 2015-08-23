from distutils.core import setup

setup(
    name='pyIID',
    version='',
    packages=['extra', 'pyiid', 'pyiid.sim', 'pyiid.calc', 'pyiid.tests',
              'pyiid.tests.test_master', 'pyiid.tests.test_master.test_sim',
              'pyiid.tests.test_master.test_calc', 'pyiid.kernels',
              'pyiid.testing', 'pyiid.experiments', 'pyiid.experiments.rixs',
              'pyiid.experiments.saxs', 'pyiid.experiments.exafs',
              'pyiid.experiments.elasticscatter',
              'pyiid.experiments.elasticscatter.mpi',
              'pyiid.experiments.elasticscatter.cpu_wrappers',
              'pyiid.experiments.elasticscatter.gpu_wrappers', 'benchmarks',
              'benchmarks.speed'],
    url='',
    license='',
    author='christopher',
    author_email='',
    description=''
)
