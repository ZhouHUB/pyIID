__author__ = 'christopher'
from pyiid.sim.nuts_hmc import nuts
import numpy as np
from ase.atoms import Atoms

from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from ase.visualize import view


def test_nuts():
    """
    Test NUTS simulation
    :return:
    """
    # ideal_atoms = Atoms('Au2', [[0, 0, 0], [3, 0, 0]])
    ideal_atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    s = ElasticScatter()
    gobs = s.get_pdf(ideal_atoms)

    ideal_atoms.positions *= 1.02

    calc = PDFCalc(obs_data=gobs, scatter=s, conv=300, potential='rw')
    ideal_atoms.set_calculator(calc)
    initial_pe = ideal_atoms.get_potential_energy()

    traj, _, _ = nuts(ideal_atoms, .65, 10, 1., tree_limit=5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    del traj
    print min_pe, initial_pe
    assert min_pe < initial_pe

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v', '--nocapture'], exit=False)