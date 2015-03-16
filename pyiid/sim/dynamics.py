__author__ = 'christopher'
import numpy as np
from copy import deepcopy as dc


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
    f = atoms.get_forces()
    atoms.set_momenta(atoms.get_momenta() + step * f)
    new_pos = atoms.get_positions() + step * atoms.get_momenta()
    atoms.set_positions(new_pos)
    return atoms


def simulate_dynamics(atoms, stepsize, n_steps):
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
    # get the initial positions and momenta
    grad = atoms.get_forces()
    moves = [dc(atoms)]
    atoms.set_momenta(atoms.get_momenta() + 0.5 * stepsize * grad)
    atoms.set_positions(
        atoms.get_positions() + stepsize * atoms.get_momenta())
    moves += [dc(atoms)]
    for n in range(n_steps):
        prop_move = [dc(leapfrog(atoms, stepsize))]
        moves += prop_move
    atoms.set_momenta(atoms.get_momenta() + 0.5 * stepsize *
                         atoms.get_forces())
    return moves
