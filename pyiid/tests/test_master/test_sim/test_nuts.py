from itertools import product
import numpy as np
from pyiid.calc.fqcalc import FQCalc
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.sim.nuts_hmc import NUTSCanonicalEnsemble
from pyiid.tests import test_atom_squares, test_calcs

__author__ = 'christopher'

test_nuts_data = tuple(product(test_atom_squares, test_calcs))


def test_nuts_dynamics():
    for v in test_nuts_data:
        yield check_nuts, v


def check_nuts(value):
    """
    Test NUTS simulation
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    if value[1] == 'PDF':
        s = ElasticScatter()
        gobs = s.get_pdf(ideal_atoms)
        calc = PDFCalc(obs_data=gobs, scatter=s, conv=300, potential='rw')
    elif value[1] == 'FQ':
        s = ElasticScatter()
        gobs = s.get_fq(ideal_atoms)
        calc = FQCalc(obs_data=gobs, scatter=s, conv=300, potential='rw')
    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    start_pe = ideal_atoms.get_potential_energy()

    # traj, _, _, _ = nuts(ideal_atoms, .65, 3, 1., escape_level=4)
    nuts = NUTSCanonicalEnsemble(ideal_atoms, escape_level=4)
    traj = nuts.run(5)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print len(traj)
    del traj
    print min_pe, start_pe
    assert min_pe < start_pe


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
