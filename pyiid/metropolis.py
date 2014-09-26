__author__ = 'christopher'
import numpy as np


def mh(atoms, iterations, translate_frequency, max_translate, swap_frequency):
    """
    The metropolis-hastings algorithm

    Parameters
    ----------
    atoms: ase.atoms object
        The system
    iterations: int
        The number of iterations to perform
    translate_frequency: float [0,1]
        The likelihood that a given iteration will include a translation
    max_translate: float
        The maximum translation of a given step in angstroms
    swap_frequency: float [0,1]
        The likelihood of a given iteration including a swap move

    Returns
    -------
    atoms:
        The final configuration
    a bunch of fit metrics
    """
    initial_nrg = []
    for potential in potentials:
        initial_nrg += calc_nrg(potential)
    i_counter = 0
    accepted = 0
    bad_accepted = 0
    while i_counter <= iterations:
        for move in moves:
            if np.random.random(1) <= move.frequency:
                new_atoms = move.perform_move(atoms)
        current_nrg = []
        for potential in potentials:
            current_nrg += calc_nrg(potential)
        if combine(current_nrg) < combine(initial_nrg):
            atoms = new_atoms
            accepted += 1
        elif #compare energy to temperature
            atoms = new_atoms
            accepted += 1
            bad_accepted +=1
    return atoms, accepted, bad_accepted, current_nrg