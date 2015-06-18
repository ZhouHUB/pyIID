__author__ = 'christopher'
import numpy as np
from copy import deepcopy as dc
from pyiid.sim import leapfrog


def classical_dynamics(atoms, stepsize, n_steps, temp=0):
    """
    Create a new atomic configuration by simulating the hamiltonian dynamics
    of the system

    Parameters
    ----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration
    stepsize: float
        The step size for the simulation
    n_steps: int
        The number of steps

    Returns
    -------
         list of ase.Atoms
        This list contains all the moves made during the simulation
    """
    f = atoms.get_forces()
    traj = [atoms]
    for n in range(n_steps):
        traj.append(leapfrog(traj[-1], stepsize))
    return traj




if __name__ == '__main__':
    import os
    from copy import deepcopy as dc
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.visualize import view
    from ase.io.trajectory import PickleTrajectory
    import ase.io as aseio
    import time
    import datetime
    import math
    from ase.atoms import Atoms

    from pyiid.wrappers.elasticscatter import ElasticScatter
    from pyiid.calc.pdfcalc import PDFCalc
    from pyiid.calc import wrap_rw
    from pyiid.utils import tag_surface_atoms, build_sphere_np, load_gr_file, \
        time_est

    # ideal_atoms = Atoms(['Sn','O','O', 'Sn'], [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms = Atoms('Sn2O2', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    ideal_atoms.set_masses([1, 1, 1, 1])
    ideal_atoms.set_velocities(np.zeros((len(ideal_atoms), 3)))
    start_atoms = dc(ideal_atoms)
    start_atoms.positions *= 1.02
    s = ElasticScatter()
    gobs = s.get_pdf(ideal_atoms)


    calc = PDFCalc(obs_data=gobs, scatter=s, conv=100, potential='rw')
    start_atoms.set_calculator(calc)
    print start_atoms.get_potential_energy()

    e = .8e-2
    M = 0

    traj= classical_dynamics(start_atoms, e, 100)

    pe_list = []
    for atoms in traj:
        pe_list.append(atoms.get_potential_energy())
        f = atoms.get_forces()
        f2 = f * 2
    min_pe = np.argmin(pe_list)
    view(traj)
    print traj[-1].get_potential_energy()
    r = np.arange(0, 40, .01)
    plt.plot(r, gobs, 'b-', label='ideal')
    plt.plot(r, s.get_pdf(traj[0]) *
             wrap_rw(s.get_pdf(traj[0]), gobs)[1], 'k-',
             label='start')
    plt.plot(r, s.get_pdf(traj[-1]) *
             wrap_rw(s.get_pdf(traj[-1]), gobs)[1], 'r-',
             label='final')
    plt.legend()
    plt.show()
