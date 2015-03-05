__author__ = 'christopher'
import ase.io as io
import numpy as np


def rmc_translate(atoms, max_translation=1):
    """
    This function performs random translations on the atoms

    Parameters
    ----------
    atoms: ase.atoms object
        The system
    max_translation: float
        The maximum translation in Angstroms

    Returns
    -------
    ase.atoms object:
        The resulting configuration
    """
    atom_count = len(atoms)
    translation = np.random.random((atom_count, 3))
    translation *= np.random.choice([-1, 1], size=(atom_count, 3))
    translation *= max_translation
    old_pos = atoms.get_positions()
    new_pos = old_pos+translation
    atoms.set_positions(new_pos)
    return atoms


def rmc_swap(atoms):
    """
    Swap the atomic elements in the system

    Parameters
    ----------
    atoms: ase.atoms object
        The system

    Returns
    -------
    ase.atoms object:
        The resulting configuration
    """
    old_numbers = atoms.get_atomic_numbers()
    new_numbers = np.random.shuffle(old_numbers)
    atoms.set_atomic_numbers(new_numbers)
    return atoms

