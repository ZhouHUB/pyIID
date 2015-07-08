__author__ = 'christopher'
from pyiid.wrappers.flat_multi_cpu_wrap import wrap_fq as ffq
from pyiid.wrappers.cpu_wrap import wrap_fq as cfq
from ddt import ddt, data, unpack
import unittest
from numpy.testing import assert_allclose
from pyiid.tests import *

test_data = ((setup_atoms(n), .1) for n in np.logspace(1, 3, 3))


@ddt
class FQTest(unittest.TestCase):
    @data(*test_data)
    def test_fq(self, value):
        rtol = 1e-7
        # The major error in our code comes from the sine function.
        # However, this error is multiplicative in the number of atoms.
        # Thus, the absolute tolerance of our sine is multiplied by the number
        # of atoms
        atol = 6e-6 * len(value[0])
        f = ffq(*value)
        c = cfq(*value)
        print len(value[0])
        i = np.where((f - c) > (rtol * np.abs(c) + atol))[0]
        print [a for a in zip(i, (f - c)[i], (rtol * np.abs(c) + atol)[i])]
        assert_allclose(f, c, rtol, atol)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
