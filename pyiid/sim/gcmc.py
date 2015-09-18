from copy import deepcopy as dc
import numpy as np
from ase.atom import Atom
from ase.units import *
from pyiid.sim import Ensemble

__author__ = 'christopher'


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
    if np.random.random((1,)) < np.exp(min([0, np.log(atoms.get_volume() / (
                len(atoms) + 1)) - beta * delta_energy + beta * mu])):
        return atoms_prime
    else:
        return None


def del_atom(atoms, chem_potentials, beta):
    if len(atoms) == 1:
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


class GrandCanonicalEnsemble(Ensemble):
    """
    Grand Canonical Monte Carlo simulation
    >>> from ase.cluster.octahedron import Octahedron
    >>> from pyiid.calc.spring_calc import Spring
    >>> atoms = Octahedron('Au', 3)
    >>> atoms.rattle(.1)
    >>> atoms.center()
    >>> calc = Spring(rt=2.5, k=200)
    >>> atoms.set_calculator(calc)
    >>> gc = GrandCanonicalEnsemble(atoms, {'Au': 0.0}, 3000)
    >>> traj = gc.run(10000)
    """

    def __init__(self, atoms, chemical_potentials, temperature=100,
                 restart=None, logfile=None, trajectory=None, seed=None):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed)
        self.beta = 1. / (temperature * kB)
        self.chem_pot = chemical_potentials

    def step(self):
        if self.random_state.uniform() >= .5:
            new_atoms = del_atom(self.traj[-1], self.chem_pot, self.beta)
        else:
            new_atoms = add_atom(self.traj[-1], self.chem_pot, self.beta)
        if new_atoms is not None:
            self.traj.append(new_atoms)
            return [new_atoms]
        else:
            return None
