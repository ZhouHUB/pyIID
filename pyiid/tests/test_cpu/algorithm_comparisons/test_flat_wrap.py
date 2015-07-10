__author__ = 'christopher'
from pyiid.tests import *

from pyiid.wrappers.cpu_wrappers.flat_multi_cpu_wrap import wrap_fq as fq1
from pyiid.wrappers.cpu_wrappers.cpu_wrap import wrap_fq as fq2

from pyiid.wrappers.cpu_wrappers.flat_multi_cpu_wrap import wrap_fq_grad as gfq1
from pyiid.wrappers.cpu_wrappers.cpu_wrap import wrap_fq_grad as gfq2


test_data = product(test_atoms, test_qbin)


@ddt
class TestWrap(TC):
    @data(*test_data)
    def test_wrap_grad_fq(self, value):
        rtol = 1e-7
        # The major error in our code comes from the sine function.
        # However, this error is multiplicative in the number of atoms.
        # Thus, the absolute tolerance of our sine is multiplied by the number
        # of atoms
        atol = 6e-6 * len(value[0])
        f = gfq1(*value)
        c = gfq2(*value)
        print len(value[0])
        print np.amax(np.abs(f - c)), np.mean(np.abs(f - c)), np.std(
            np.abs(f - c))
        assert_allclose(f, c, rtol, atol)

@ddt
class TestWrap2(TC):
    @data(*test_data)
    def test_wrap_fq(self, value):
        rtol = 1e-7
        # The major error in our code comes from the sine function.
        # However, this error is multiplicative in the number of atoms.
        # Thus, the absolute tolerance of our sine is multiplied by the number
        # of atoms
        atol = 6e-6 * len(value[0])
        f = fq1(*value)
        c = fq2(*value)
        print len(value[0])
        i = np.where((f - c) > (rtol * np.abs(c) + atol))[0]
        print [a for a in zip(i, (f - c)[i], (rtol * np.abs(c) + atol)[i])]
        assert_allclose(f, c, rtol, atol)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
