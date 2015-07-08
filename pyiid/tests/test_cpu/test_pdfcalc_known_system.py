__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.tests import *

test_atoms = [setup_atomic_square()]
test_exp = [generate_experiment() for i in range(3)]

test_data = tuple(product(test_atoms, test_exp))

proc = 'CPU'
alg = 'flat'
alg1 = 'flat'
alg2 = 'nxn'

@ddt
class TestScatter(TC):
    """
    Test flat cpu scatter
    """
    @data(*test_data)
    def test_rw(self, value):
        """
        Test two random systems against one another for Rw
        """
        atoms1, atoms2 = value[0]
        scat = ElasticScatter()
        scat.set_processor(proc, alg)
        gobs = scat.get_pdf(atoms1)
        calc = PDFCalc(obs_data=gobs, scatter=scat, potential='rw')
        atoms2.set_calculator(calc)
        rw = atoms2.get_potential_energy()
        # print rw
        assert rw >= .9


def test_chi_sq():
    """
    Test two random systems against one another for $\chi^{2}$
    """
    atoms1, atoms2, scale = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential='chi_sq')
    atoms2.set_calculator(calc)
    chi_sq = atoms2.get_potential_energy()
    # print chi_sq
    assert chi_sq >= 1000
    # assert False


def test_grad_rw():
    """
    Test two random systems against one another for grad rw
    """
    atoms1, atoms2, scale = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    scat.set_processor('CPU')
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential='rw')
    atoms2.set_calculator(calc)
    forces = atoms2.get_forces()
    com = atoms2.get_center_of_mass()
    for i in range(len(atoms2)):
        dist = atoms2[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))


def test_grad_chi_sq():
    """
    Test two random systems against one another for grad $\chi^{2}$
    """
    atoms1, atoms2, scale = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(obs_data=gobs, scatter=scat, potential='chi_sq')
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
