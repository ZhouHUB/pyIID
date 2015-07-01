from copy import deepcopy as dc
from ase.units import kB
import numpy as np

__author__ = 'christopher'


def leapfrog(atoms, step):
    """
    Propagate the dynamics of the system via the leapfrog algorithm one step

    Parameters
    -----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration for the system
    step: float
        The step size for the simulation, the new momentum/velocity is step *
        the force

    Returns
    -------
    ase.Atoms
        The new atomic positions and velocities
    """
    latoms = dc(atoms)

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())

    latoms.positions += step * latoms.get_velocities()

    latoms.set_momenta(latoms.get_momenta() + 0.5 * step * latoms.get_forces())

    return latoms


def set_momentum_from_temperature(atoms, temp):
    e_num = atoms.get_atomic_numbers()
    e_set = set(e_num)
    e_list = list(e_set)
    e_mass = np.asarray(list(set(atoms.get_masses())))
    e_prms = np.sqrt(3 * kB * temp * e_mass)
    p = np.zeros((len(atoms), 3))
    for i in range(len(e_set)):
        in_set = np.where(atoms.numbers == e_list[i])[0]
        p[in_set, :] = np.random.normal(0, e_prms[i], (len(in_set), 3))
    atoms.set_momenta(p)


if __name__ == '__main__':
    from ase.atoms import Atoms

    atoms = Atoms('Sn2O2', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    set_momentum_from_temperature(atoms, 300)
    print np.random.normal(0, [1], 3)
    print atoms.get_momenta()
