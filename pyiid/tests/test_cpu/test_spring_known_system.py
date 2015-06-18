__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.spring_calc import Spring
from pyiid.tests import setup_atomic_configs


def test_spring():
    """
    Test two random systems against one another for Rw
    """
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=5.)
    atoms2.set_calculator(calc)
    assert atoms2.get_potential_energy() >= 100


def test_grad_spring():
    """
    Test two random systems against one another for grad rw
    """
    atoms1, atoms2, scale = setup_atomic_configs()
    scat = ElasticScatter()
    scat.set_processor('CPU')
    scat.set_processor('CPU')
    calc = Spring(k=100, rt=5.)
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
