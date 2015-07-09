__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.tests import *

test_atoms = [setup_atomic_square()]
test_exp = [None]
test_exp.extend([generate_experiment() for i in range(3)])

test_potentials = [('rw', .9), ('chi_sq', 500)]

test_data = tuple(product(test_atoms, test_exp, test_potentials))
proc = 'CPU'
alg1 = 'flat'
alg2 = 'nxn'

alg = alg2


@ddt
class TestPDFCalc(TC):
    """
    Test flat cpu scatter
    """
    @data(*test_data)
    def test_nrg(self, value):
        """
        Test two random systems against one another for Rw
        """
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        scat.update_experiment(exp_dict=value[1])
        scat.set_processor(proc, alg)
        p, thresh = value[2]

        gobs = scat.get_pdf(atoms1)
        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)

        ans = atoms2.get_potential_energy()
        assert ans >= thresh

    @data(*test_data)
    def test_forces(self, value):
        """
        Test two random systems against one another for Rw
        """
        # setup
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        scat.update_experiment(exp_dict=value[1])
        scat.set_processor(proc, alg)
        p, thresh = value[2]

        gobs = scat.get_pdf(atoms1)
        calc = PDFCalc(obs_data=gobs, scatter=scat, potential=p)
        atoms2.set_calculator(calc)

        forces = atoms2.get_forces()
        com = atoms2.get_center_of_mass()
        for i in range(len(atoms2)):
            dist = atoms2[i].position - com
            # print i, dist, forces[i], np.cross(dist, forces[i])
            assert_allclose(np.cross(dist, forces[i]), np.zeros(3))

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
