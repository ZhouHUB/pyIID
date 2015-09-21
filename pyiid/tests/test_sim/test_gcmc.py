from ase.cluster import FaceCenteredCubic
from pyiid.sim.gcmc import GrandCanonicalEnsemble
from pyiid.calc.spring_calc import Spring
from pyiid.tests import *

__author__ = 'christopher'


def test_negative_mu():
    atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (3, 6, 3))
    calc = Spring(rt=2.5, k=200)
    atoms.set_calculator(calc)
    gce = GrandCanonicalEnsemble(atoms, {'Au': -100})
    traj = gce.run(100)
    tf = False
    for i in range(len(traj)):
        if len(traj[i]) < len(traj[0]):
            tf = True
            break
    assert tf


def test_positive_mu():
    atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (3, 6, 3))
    calc = Spring(rt=2.5, k=200)
    atoms.set_calculator(calc)
    gce = GrandCanonicalEnsemble(atoms, {'Au': 100})
    traj = gce.run(100)
    tf = False
    for i in range(len(traj)):
        if len(traj[i]) > len(traj[0]):
            tf = True
            break
    assert tf

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         # '-v',
                         # '-x'
                         ],
                   exit=False)

