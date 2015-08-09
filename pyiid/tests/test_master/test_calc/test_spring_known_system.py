__author__ = 'christopher'
from pyiid.tests import *
import numpy as np
from pyiid.calc.spring_calc import Spring


def test_spring():
    """
    Test spring for atomic square
    """
    atoms1, _ = setup_atomic_square()
    calc = Spring(k=100, rt=5.)
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring():
    """
    Test gradient of spring for atomic square
    """
    atoms1, _ = setup_atomic_square()
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
    Test center of mass attractive spring
    """
    atoms1, _= setup_atomic_square()
    calc = Spring(k=100, rt=1., sp_type='com')
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring2():
    """
    Test center of mass attractive spring gradient
    """
    atoms1, _ = setup_atomic_square()
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
    Test pair attractive spring
    """
    atoms1, _ = setup_atomic_square()
    calc = Spring(k=100, rt=1., sp_type='att')
    atoms1.set_calculator(calc)
    assert atoms1.get_potential_energy() >= 100


def test_grad_spring3():
    """
    Test pair attractive spring gradient
    """
    atoms1, _ = setup_atomic_square()
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
