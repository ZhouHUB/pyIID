from pyiid.sim import leapfrog
__author__ = 'christopher'


def classical_dynamics(atoms, stepsize, n_steps, stationary=False,
                       zero_rotation=False):
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
    stationary: bool
        Whether to keep the atoms stationary
    zero_rotation: bool
        Whether to prevent configuration rotation

    Returns
    -------
         list of ase.Atoms
        This list contains all the moves made during the simulation
    """
    atoms.get_forces()
    traj = [atoms]
    for n in range(n_steps):
        traj.append(leapfrog(traj[-1], stepsize, stationary, zero_rotation))
    return traj
