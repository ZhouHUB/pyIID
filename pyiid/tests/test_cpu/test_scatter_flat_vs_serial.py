__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.tests import *
from numba import cuda

test_atoms = [setup_atoms(n) for n in np.logspace(1, 3, 3)]
test_exp = [None]
test_exp.extend([generate_experiment() for i in range(3)])

test_data = tuple(product(test_atoms, test_exp))

proc = 'CPU'
alg1 = 'flat'
alg2 = 'nxn'


@ddt
class TestScatter(TC):
    """
    Test flat cpu scatter
    """

    @data(*test_data)
    def test_scatter_fq(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_fq(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_fq(atoms)

        # test
        print np.max(np.abs(ans1 - ans2)), np.mean(
            np.abs(ans1 - ans2)), np.std(np.abs(ans1 - ans2))
        assert_allclose(ans1, ans2, atol=atol)

    @data(*test_data)
    def test_scatter_sq(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_sq(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_sq(atoms)

        # test
        assert_allclose(ans1, ans2, rtol=1e-3, atol=atol)

    @data(*test_data)
    def test_scatter_iq(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_iq(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_iq(atoms)

        # test
        assert_allclose(ans1, ans2, rtol=1e-3, atol=atol)

    @data(*test_data)
    def test_scatter_pdf(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_pdf(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_pdf(atoms)

        # test
        assert_allclose(ans1, ans2, atol=atol)

    @data(*test_data)
    def test_scatter_grad_fq(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_grad_fq(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_grad_fq(atoms)

        # test
        assert_allclose(ans1, ans2, atol=atol)

    @data(*test_data)
    def test_scatter_grad_pdf(self, value):
        # set everything up
        atoms, exp = value
        atol = 6e-6 * len(atoms)
        scat = ElasticScatter(exp_dict=exp)

        # run algorithm 1
        scat.set_processor(proc, alg1)
        ans1 = scat.get_grad_pdf(atoms)

        # run algorithm 2
        scat.set_processor(proc, alg2)
        ans2 = scat.get_grad_pdf(atoms)

        # test
        assert_allclose(ans1, ans2, atol=atol)

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
