__author__ = 'christopher'
import time

import numpy as np
import ase.io as aseio

from old_files.old_hmc.potential_core import Debye_srreal_U
import copy.deepcopy as dc
from pyiid.utils import load_gr_file


@autojit(target='gpu')
def mc_grad(atoms, exp_data, U, delta_qi, fixed = (None,)):
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
        if atom.tag not in fixed:
            for i in range(0, 3):
                new_atoms = dc(atoms)
                new_atoms[atom.index].position[i] += delta_qi
                new_u = U(new_atoms, exp_data)
                grad[atom.index, i] = (new_u-current_U)/delta_qi
            # print grad[atom.index]
    return grad

if __name__ == '__main__':

    atoms = aseio.read('../../../IID_data/mhmc_NiPd_25nm.traj')
    exp_data = load_gr_file("../../../IID_data/7_7_7_FinalSum.gr")
    t0 = time.clock()
    grad = mc_grad(atoms, exp_data, Debye_srreal_U, 10**-6)
    t1 = time.clock()
    delta_t = t1-t0
    print(delta_t)
    print(grad)