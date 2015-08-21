__author__ = 'christopher'
from copy import deepcopy as dc
import numpy as np
from ase.atom import Atom


def add_atom(atoms, chem_potentials, beta):
    atoms.center()
    # make the proposed system
    atoms_prime = dc(atoms)
    # make new atom
    new_symbol = np.random.choice(atoms.get_chemical_symbols())
    new_positions = np.random.uniform(0, np.max(atoms.get_cell(), 0))
    new_atom = Atom(new_symbol, np.asarray(new_positions))

    # append new atom to system
    atoms_prime.append(new_atom)

    # get new energy
    delta_energy = atoms_prime.get_total_energy() - atoms.get_total_energy()
    # get chemical potential
    mu = chem_potentials[new_symbol]
    # calculate acceptance
    if np.random.random((1,)) < np.exp(min([0, np.log(atoms.get_volume() / (len(atoms) + 1)) - beta * delta_energy +
                    beta * mu
            ])):
        return atoms_prime
    else:
        return None


def del_atom(atoms, chem_potentials, beta):
    if len(atoms) ==1:
        return None
    atoms.center()
    # make the proposed system
    atoms_prime = dc(atoms)
    del_atom_index = np.random.choice(range(len(atoms)))
    del_symbol = atoms_prime[del_atom_index].symbol

    # append new atom to system
    del atoms_prime[del_atom_index]

    # get new energy
    delta_energy = atoms_prime.get_total_energy() - atoms.get_total_energy()
    # get chemical potential
    mu = chem_potentials[del_symbol]
    # calculate acceptance
    if np.random.random() < np.exp(min([0,
                                        np.log(len(atoms) / atoms.get_volume())
                                                - beta * delta_energy
                                                - beta * mu
                                        ])):
        return atoms_prime
    else:
        return None


def grand_cannonical_move(traj, chem_potentials, beta):
    rand = np.random.random()
    if rand >= .5:
        new_atoms = del_atom(traj[-1], chem_potentials, beta)
    else:
        new_atoms = add_atom(traj[-1], chem_potentials, beta)
    if new_atoms is not None:
        traj.append(new_atoms)


if __name__ == '__main__':
    from ase.cluster.octahedron import Octahedron
    from pyiid.calc.spring_calc import Spring
    import matplotlib.pyplot as plt
    from ase.visualize import view

    atoms = Octahedron('Au', 3)
    atoms.center()
    calc = Spring(
        rt=20,
        sp_type='att'
    )
    atoms.set_calculator(calc)
    traj = [atoms]
    n = []
    for i in range(10000):
        grand_cannonical_move(traj, {'Au': -4.25}, 1)
        n.append(len(traj[-1]))
    plt.plot(n)
    plt.show()
    # view(traj[-1])