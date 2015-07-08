__author__ = 'christopher'

from numba import *
import numpy as np
from numpy.testing import assert_allclose
import math
from pyiid.wrappers import generate_grid
import matplotlib.pyplot as plt
from ddt import data, ddt
import unittest

@cuda.jit(argtypes=[f4[:], f4[:]])
def gpu_sin(dout, din):
    i = cuda.grid(1)
    if i >= len(din):
        return
    dout[i] = math.sin(din[i])

@ddt
class TestCase(unittest.TestCase):
    @data(1000, 2000, 10)
    def test(self, value):
        data = np.linspace(0, 3*np.pi, value, dtype=np.float32)
        cpu = np.sin(data)
        tpb = [64]
        bpg = generate_grid(data.shape, tpb)
        din = cuda.to_device(data)
        out = np.zeros(data.shape, np.float32)
        dout = cuda.device_array(data.shape, dtype=np.float32)
        gpu_sin[bpg, tpb](dout, din)
        dout.copy_to_host(out)
        # assert False
        assert_allclose(cpu, out, 1e-6)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)