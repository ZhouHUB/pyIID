from ase.cluster import FaceCenteredCubic
from pyiid.sim.gcmc import GCEnsemble
from pyiid.calc.spring_calc import Spring
from pyiid.tests import *

__author__ = 'christopher'


def test_negative_mu():
    atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (3, 6, 3))
    calc = Spring(rt=2.5, k=200)
    atoms.set_calculator(calc)
    gce = GCEnsemble(atoms, {'Au': -100})
    traj = gce.run(1000)
    assert len(traj[-1]) == 1


def test_positive_mu():
    atoms = FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (3, 6, 3))
    calc = Spring(rt=2.5, k=200)
    atoms.set_calculator(calc)
    gce = GCEnsemble(atoms, {'Au': 100})
    traj = gce.run(1000)
    assert len(traj[-1]) > 100

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         # '-v',
                         # '-x'
                         ],
                   exit=False)

