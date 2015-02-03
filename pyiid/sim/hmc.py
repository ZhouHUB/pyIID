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
    atoms.set_velocities(atoms.get_velocities() + 0.5 * stepsize * grad)
    atoms.set_positions(
        atoms.get_positions() + stepsize * atoms.get_velocities())

    #do n leapfrog steps
    for n in range(n_steps):
        prop_move = leapfrog(atoms, stepsize)

    atoms.set_velocities(
        atoms.get_velocities() + 0.5 * stepsize * atoms.get_forces())


def mh_accept(initial_energy, next_energy, T=1):
    """
    Decide whether to accept or reject the new configuration

    Parameters
    -----------
    initial_energy: float
        The Hamiltonian of the intial configuration
    next_energy: float
        The Hamiltonian of the proposed configuration

    Returns
    -------
    np._bool
        Whether to accept the new configuration
    """
    diff = initial_energy - next_energy
    canon_part_prob = np.exp(diff * T)

    rand = np.random.random((1,))

    print 'initial, next, rand, exp'
    print initial_energy, next_energy, rand, canon_part_prob

    return rand < canon_part_prob


def hmc_move(atoms, stepsize, n_steps, T):
    """
    Move atoms and check if they are accepted, returning the new 
    configuration if accepted, the old if not
    
    Parameters
    -----------
    atoms: ase.Atoms 
    stepsize: 
    n_steps:
    
    Returns
    --------
    bool
        True if new atoms accepted, false otherwise
    ase.Atoms
        The atomic configuration to be used in the next iteration
    
    """
    atoms.set_velocities(
        np.random.normal(0, 1, (len(atoms), 3)) / 3 / len(atoms))
    prop = dc(atoms)

    simulate_dynamics(prop, stepsize, n_steps)

    accept = mh_accept(atoms.get_total_energy(), prop.get_total_energy(), T)

    if accept == 1:
        return True, prop
    elif accept == 0:
        return False, atoms


def run_hmc(atoms, iterations, stepsize, n_steps, avg_acceptance_slowness,
            avg_acceptance_rate, target_acceptance_rate, stepsize_inc,
            stepsize_dec, stepsize_min, stepsize_max, T=1):
    """
    Wrapper for running Hamiltonian (Hybrid) Monte Carlo refinements,
    using a dynamic step size refinement, based on whether moves are
    being accepted
    
    Parameters
    ----------
    atoms: ase.Atoms
    iterations: int
        The number of
    stepsize: float
        The stepsize for the simulation
    n_steps: int
        The number of steps to take during the simulation
    avg_acceptance_slowness: float
        The desired time constant
    avg_acceptance_rate: float
    target_acceptance_rate: float
        The desired acceptance rate
    stepsize_inc: float
        The amount by which the stepsize is increased
    stepsize_dec: float
        The amount by which the stepsize is decreased
    stepsize_min: float
        The minimum stepsize
    stepsize_max: float
        The maximum stepsize
    T: float
        The simulation temperature

    Returns
    -------
    traj: list of ase.Atoms
        The list of configurations
    accept_list: list of bools
        The list describing whether moves were accpeted or rejected
    """

    # initialize lists for holding interesting variables
    traj = [atoms]
    accept_list = []

    i = 0
    try:
        while i < iterations:
            accept, atoms = hmc_move(atoms, stepsize, n_steps, T)
            print i, accept, atoms.get_potential_energy(), stepsize
            print '--------------------------------'
            if accept is True:
                traj += [atoms]
            accept_list.append(accept)
            avg_acceptance_rate = avg_acceptance_slowness \
                                  * avg_acceptance_rate \
                                  + (1.0 - avg_acceptance_slowness) \
                                    * np.average(np.array(accept_list))

            if avg_acceptance_rate > target_acceptance_rate:
                stepsize *= stepsize_inc
            else:
                stepsize *= stepsize_dec
            if stepsize > stepsize_max:
                stepsize = stepsize_max
            elif stepsize < stepsize_min:
                stepsize = stepsize_min
            i += 1
    except KeyboardInterrupt:
        pass
    return traj, accept_list