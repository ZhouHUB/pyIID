from pyiid.tests import *
from pyiid.sim.dynamics import classical_dynamics
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.calc.fqcalc import FQCalc
from ase.visualize import view

__author__ = 'christopher'

test_dynamics_data = tuple(product(test_atom_squares, test_calcs, [1, -1],
                                   (True, False), (True, False)))


def test_gen_dynamics():
    for v in test_dynamics_data:
        yield check_dynamics, v


def check_dynamics(value):
    """
    Test classical dynamics simulation, symplectic dynamics are look the same
    forward as reversed
    """
    ideal_atoms, _ = value[0]
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))

    if value[1] == 'PDF':
        s = ElasticScatter()
        gobs = s.get_pdf(ideal_atoms)
        calc = PDFCalc(obs_data=gobs, scatter=s, conv=3, potential='rw')

    elif value[1] == 'FQ':
        s = ElasticScatter()
        gobs = s.get_fq(ideal_atoms)
        calc = FQCalc(obs_data=gobs, scatter=s, conv=3, potential='rw')

    else:
        calc = value[1]
    ideal_atoms.positions *= 1.02

    ideal_atoms.set_calculator(calc)
    start_pe = ideal_atoms.get_potential_energy()
    e = value[2]
    traj = classical_dynamics(ideal_atoms, e, 5, value[3], value[4])

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
    min_pe = np.min(pe_list)
    print min_pe, start_pe, len(traj)
    print pe_list
    assert min_pe < start_pe
    if value[3]:
        assert_allclose(traj[-1].get_center_of_mass(),
                        traj[0].get_center_of_mass())


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['--with-doctest',
                         # '--nocapture',
                         '-v',
                         '-x'
                         ],
                   exit=False)
