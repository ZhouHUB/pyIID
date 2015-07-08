__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.tests import *
from numba import cuda
from itertools import *

test_atoms = [setup_atoms(n) for n in np.logspace(1, 3, 3)]
test_exp = [generate_experiment() for i in range(3)]

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
        assert_allclose(ans1, ans2, atol=atol)

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
        assert_allclose(ans1, ans2, rtol=1e-5, atol=atol)

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

'''
    @data(*test_data)
    def test_scatter_grad_fq(self, value):
        """
        Smoke test CPU Grad F(Q) from scatter
        """

        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, proc)
        # Test a set of different sized ensembles
        ans = scat.get_grad_fq(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)



    @data(*test_data)
    def test_scatter_grad_pdf(self, value):
        """
        Smoke test CPU Grad PDF from scatter
        """

        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, proc)
        # Test a set of different sized ensembles
        ans = scat.get_grad_pdf(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)
'''

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
