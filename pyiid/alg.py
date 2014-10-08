__author__ = 'christopher'

import numpy as np
from copy import deepcopy as dc
from ase import Atoms
from potential_core import Debye_srfit_U


def PDF_only_grad(atoms, exp_data, delta_qi, current_U=None):
    """
    Calculate the Gradient for only the PDF potential

    Parameters
    ----------
    atoms: ase.atoms
        The atomic configuration
    exp_data: ndarray
        The experimental PDF, or other data used in U calculation
    current_U:float
        The current potential energy
    delta_qi: float
        The size of the change in position for calculation of the grad

    Returns
    -------
    ndarray:
        The gradient
    """
    grad = np.zeros(len(atoms), 3)
    q = atoms.get_positions()
    for i in range(q.shape()[0]):
        for j in range(0, 3):
            new_q = q
            new_q[i, j] += delta_qi


def HMC(current_atoms, U, current_U, grad_U, exp_data, T, epsilon, L,
        delta_qi, fixed=(None,)):
    """
    Hamiltonian monte carlo engine

    Parameters
    ----------
    current_atoms: ase.atoms
        Starting atomic configuration
    exp_data: ndarray
        The experimental PDF, or other data used in U calculation
    U: func
        The potential energy function
    grad_U: func
        The function for gradian calculation
    exp_data: ndarray
        Experimental data to be fed to the potential energy calculatior
    T: float
        The temperature, higher temps mean more bad moves accepted
    epsilon: float, maybe int
        The leapfrog step size
    L: int
        The number of steps before calcualting U and K

    Returns
    -------
    atoms, bool, float, float:
        The configuration, if a move was bad, the U and K of the configuration

    """
    q = current_atoms.get_positions()
    p = np.random.normal(0, .1, q.shape)
    current_p = p
    p -= epsilon * grad_U(current_atoms, exp_data, U, delta_qi, fixed)\
         / 2
    # print p
    # print grad_U(current_atoms, exp_data, U, delta_qi)
    prop_atoms = dc(current_atoms)
    for i in range(L):
        q += epsilon * p
        prop_atoms.set_positions(q)
        if i != L:
            # Uq = U(prop_atoms, exp_data)
            p -= epsilon * grad_U(prop_atoms, exp_data, U, delta_qi)
    p -= epsilon * grad_U(prop_atoms, exp_data, U, delta_qi)/2
    p = -p
    #maybe pull this out and take in from previous run
    current_U = U(current_atoms, exp_data)
    current_K = np.sum(current_p ** 2) / 2.
    prop_U = U(prop_atoms, exp_data)
    prop_K = np.sum(p ** 2) / 2.
    # print(current_U-prop_U+current_K-prop_K)
    if np.random.random((1,)) * 1/T < np.exp((
                                    current_U - prop_U + current_K - prop_K)):
        if prop_U + prop_K < current_U + current_K:
            return prop_atoms, True, prop_U, prop_K
        else:
            return prop_atoms, False, prop_U, prop_K

    else:
        return current_atoms, False, current_U, current_K


def MHMC(current_atoms, U, current_U, exp_data, T, step_size, rmax, rmin,
         fixed=None):
    """Metropolis Hastings monte carlo engine

    Parameters
    ----------
    current_atoms: ase.atoms
        Starting atomic configuration
    exp_data: ndarray
        The experimental PDF, or other data used in U calculation
    U: func
        The potential energy function
    exp_data: ndarray
        Experimental data to be fed to the potential energy calculatior
    T: float
        The temperature, higher temps mean more bad moves accepted
    step_size: float
        The std distance an atom can move per step

    Returns
    -------
    atoms, bool, float, float:
        The configuration, if a move was bad, the U and K of the configuration

    """

    delta_q = np.random.normal(0, step_size, (len(current_atoms), 3))
    prop_atoms = dc(current_atoms)
    if fixed is not None:
        for a in fixed:
            delta_q[[atom.index for atom in prop_atoms if atom.tag == a]] = \
                np.zeros((3,))

    prop_atoms.translate(delta_q)
    prop_U = U(prop_atoms, exp_data, rmax, rmin)
    if prop_U < current_U:
        return prop_atoms, True, prop_U
    elif np.random.random((1,)) * 1/T < np.exp((current_U - prop_U)):
        return prop_atoms, False, prop_U
    else:
        return current_atoms, (False, False), current_U


def srFit_mhmc(atom_len, current_U, fit, T, step_size, U = Debye_srfit_U):
    delta_q = np.random.normal(0, step_size, (atom_len, 3))
    old_q = fit._contributions['NiPd'].NiPd.phase.stru.xyz
    fit._contributions['NiPd'].NiPd.phase.stru.xyz +=delta_q
    prop_U = U(fit)
    if prop_U < current_U:
        return fit, True, prop_U
    elif T <= 0:
        fit._contributions['NiPd'].NiPd.phase.stru.xyz = old_q
        return fit, (False, False), current_U
    elif np.random.random((1,)) * 1/T < np.exp((current_U - prop_U)):
        return fit, False, prop_U
    else:
        fit._contributions['NiPd'].NiPd.phase.stru.xyz = old_q
        return fit, (False, False), current_U