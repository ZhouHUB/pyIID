__author__ = 'christopher'
from pyiid.wrappers.flat_multi_cpu_wrap import wrap_fq as ffq
from pyiid.wrappers.cpu_wrap import wrap_fq as cfq
from pyiid.wrappers.flat_multi_cpu_wrap import wrap_fq_grad as ffqg
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as cfqg
from pyiid.tests import *

test_atoms = [setup_atoms(n) for n in np.logspace(1, 3, 3)]
test_qbin = [.1]
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
        f = ffqg(*value)
        c = cfqg(*value)
        print len(value[0])
        print np.amax(np.abs(f - c)), np.mean(np.abs(f - c)), np.std(
            np.abs(f - c))
        assert_allclose(f, c, rtol, atol)

    @data(*test_data)
    def test_wrap_fq(self, value):
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

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
