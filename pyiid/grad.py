__author__ = 'christopher'
import numpy as np
from copy import deepcopy as dc


def mc_grad(atoms, exp_data, U, delta_qi):
    """
    Calculate the gradient of the potential

    Parameters
    ----------
    atoms: ase.atoms
        The atomic configuration
    exp_data: ndarray
        The experimental PDF, or other data used in U calculation
    U: func
        The potential energy function
    current_U:float
        The current potential energy
    delta_qi: float
        The size of the change in position for calculation of the grad

    Returns
    -------
    ndarray:
        The gradient
    """
    current_U = U(atoms, exp_data)
    grad = np.zeros((len(atoms), 3))
    for atom in atoms:
        for i in range(0, 3):
            new_atoms = dc(atoms)
            new_atoms[atom.index].position[i] += delta_qi
            new_u = U(new_atoms, exp_data)
            grad[atom.index, i] = (new_u-current_U)/delta_qi
        # print grad[atom.index]
    return grad