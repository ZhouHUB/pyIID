__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.tests import *
from numba import cuda
from ddt import ddt, data
from itertools import *
proc = 'CPU'
alg = 'nxn'

test_atoms = [setup_atoms(n) for n in np.logspace(1, 2, 2)]
test_exp = [None]
test_exp.extend([generate_experiment() for i in range(3)])
test_data = tuple(product(test_atoms, test_exp))


@ddt
class TestScatter(TC):
    """
    Test flat cpu scatter
    """
    @data(*test_exp)
    def test_scatter_vector(self, value):
        exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        exp = scat.exp
        assert np.all(scat.get_scatter_vector() == np.arange(exp['qmin'], exp['qmax'], exp['qbin']))

    @data(*test_data)
    def test_scatter_fq(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_fq(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

    @data(*test_data)
    def test_scatter_sq(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_sq(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

    @data(*test_data)
    def test_scatter_iq(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_iq(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

    @data(*test_data)
    def test_scatter_pdf(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_pdf(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

    @data(*test_data)
    def test_scatter_grad_fq(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_grad_fq(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

    @data(*test_data)
    def test_scatter_grad_pdf(self, value):
        atoms, exp = value
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor(proc, alg)
        # Test a set of different sized ensembles
        ans = scat.get_grad_pdf(atoms)
        # Check that Scatter gave back something
        assert ans is not None
        # Check that all the values are not zero
        assert np.any(ans)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
