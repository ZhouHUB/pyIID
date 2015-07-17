__author__ = 'christopher'

from pyiid.sim import leapfrog
from pyiid.tests import setup_atomic_square
from pyiid.calc.spring_calc import Spring
import numpy as np
from numpy.testing import assert_allclose


def test_leapfrog_no_momentum():
    atoms, _ = setup_atomic_square()
    calc = Spring(rt=1, k=100)
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1)
    assert_allclose(atoms.positions, atoms2.positions)


def test_leapfrog_momentum():
    atoms, _ = setup_atomic_square()
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1)
    assert_allclose(atoms.positions, atoms2.positions - atoms.get_velocities())


def test_leapfrog_reversibility():
    atoms, _ = setup_atomic_square()
    calc = Spring(rt=1, k=100)
    atoms.set_momenta(np.ones((len(atoms), 3)))
    atoms.set_calculator(calc)
    atoms2 = leapfrog(atoms, 1)
    atoms3 = leapfrog(atoms2, -1)
    assert_allclose(atoms.positions, atoms3.positions)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
