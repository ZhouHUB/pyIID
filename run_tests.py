__author__ = 'christopher'
import nose
from pyiid.testing.noseclasses import KnownFailure
from numba import cuda
try:
    cuda.get_current_device()
    gpu = True
except:
    gpu = False

plugins = [KnownFailure]
env = {"NOSE_WITH_COVERAGE": 1,
       'NOSE_COVER_PACKAGE': 'pyiid',
       'NOSE_COVER_HTML': 1,
       'NOSE_VERBOSE': 2}

if gpu is False:
    env['NOSE_EXCLUDE_DIRS'] = 'old_files/;pyiid/tests/test_gpu'
else:
    env['NOSE_EXCLUDE_DIRS'] = 'old_files/'

from nose.plugins import multiprocess
multiprocess._instantiate_plugins = plugins


def run():

    nose.main(addplugins=[x() for x in plugins], env=env)


if __name__ == '__main__':
    run()