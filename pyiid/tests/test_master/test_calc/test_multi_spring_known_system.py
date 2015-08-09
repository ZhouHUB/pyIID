from pyiid.tests import *
import numpy as np
from pyiid.calc.spring_calc import Spring
from pyiid.calc.multi_calc import MultiCalc
__author__ = 'christopher'


def test_spring():
    """
    Test two known square systems
    """
    atoms1, atoms2 = setup_atomic_square()
    calc = MultiCalc(calc_list=[Spring(k=100, rt=5.), Spring(k=100, rt=5.)])
    atoms2.set_calculator(calc)
    assert atoms2.get_potential_energy() >= 100


def test_grad_spring():
    """
    Test two square systems
    """
    atoms1, atoms2 = setup_atomic_square()
    calc = MultiCalc(calc_list=[Spring(k=100, rt=5.), Spring(k=100, rt=5.)])
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