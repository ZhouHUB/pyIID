from ase.cluster import FaceCenteredCubic
from pyiid.tests import *
from pyiid.sim.multi_canon import MultiCanonicalSimulation
from pyiid.calc.spring_calc import Spring
from pyiid.calc.multi_calc import MultiCalc
from pyiid.sim.gcmc import GrandCanonicalEnsemble
from pyiid.sim.nuts_hmc import NUTSCanonicalEnsemble
import matplotlib.pyplot as plt
from ase.visualize import view

__author__ = 'christopher'


def test_multi_canon():
    atoms = Atoms(FaceCenteredCubic('Au', [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                              (3, 6, 3)))
    atoms.center(3)
    calc1 = Spring(rt=3., k=2000)
    calc2 = Spring(rt=3.2, k=2, sp_type='att')
    calc = MultiCalc(calc_list=[
        calc1,
        calc2
    ])
    atoms.set_calculator(calc)
    gce = GrandCanonicalEnsemble(atoms, {'Au': 0})
    nuts = NUTSCanonicalEnsemble(atoms, escape_level=4)
    sim = MultiCanonicalSimulation(atoms, [gce, nuts])
    traj = sim.run(10)
    pe = [atoms.get_potential_energy() for atoms in traj]
    n = [len(atoms) for atoms in traj]
    traj[-1].get_forces()
    view(traj[-1])
    plt.plot(pe)
    plt.show()
    plt.plot(n)
    plt.show()
    assert traj[0].get_potential_energy() > traj[-1].get_potential_energy()
    assert len(traj[0]) != len(traj[-1])

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
