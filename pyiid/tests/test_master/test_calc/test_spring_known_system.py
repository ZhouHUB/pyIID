__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.spring_calc import Spring
from pyiid.tests import setup_atomic_square


def test_spring():
    """
    Test two random systems against one another for Rw
    """
    atoms1, _ = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=5.)
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring():
    """
    Test two random systems against one another for grad rw
    """
    atoms1, _ = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=5.)
    atoms1.set_calculator(calc)
    forces = atoms1.get_forces()
    com = atoms1.get_center_of_mass()
    for i in range(len(atoms1)):
        dist = atoms1[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))


def test_spring2():
    """
    Test two random systems against one another for Rw
    """
    atoms1, _= setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=1., sp_type='com')
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring2():
    """
    Test two random systems against one another for grad rw
    """
    atoms1, _ = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=1., sp_type='com')
    atoms1.set_calculator(calc)
    forces = atoms1.get_forces()
    com = atoms1.get_center_of_mass()
    for i in range(len(atoms1)):
        dist = atoms1[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))

def test_spring3():
    """
    Test two random systems against one another for Rw
    """
    atoms1, _ = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=1., sp_type='att')
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring3():
    """
    Test two random systems against one another for grad rw
    """
    atoms1, _ = setup_atomic_square()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=1., sp_type='att')
    atoms1.set_calculator(calc)
    forces = atoms1.get_forces()
    com = atoms1.get_center_of_mass()
    for i in range(len(atoms1)):
        dist = atoms1[i].position - com
        # print i, dist, forces[i], np.cross(dist, forces[i])
        assert_allclose(np.cross(dist, forces[i]), np.zeros(3))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
