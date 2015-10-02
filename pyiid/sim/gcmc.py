from copy import deepcopy as dc
from time import time

import numpy as np
from ase.atom import Atom
from ase.units import *

from pyiid.sim import Ensemble

__author__ = 'christopher'


def add_atom(atoms, chem_potentials, beta, scatter=None):
    atoms.center()
    # make the proposed system
    atoms_prime = dc(atoms)
    # make new atom
    new_symbol = np.random.choice(atoms.get_chemical_symbols())
    if scatter is None:
        new_position = np.random.uniform(0, np.max(atoms.get_cell(), 0))
    else:
        voxels = scatter.get_total_3d_overlap(atoms)
        # choose weighted position in voxels
        qvr = np.random.choice(voxels.size, p=voxels.ravel())
        qv = np.asarray(np.unravel_index(qvr, voxels.shape))
        # put attom at center of voxel
        new_position = (qv + .5) * s.exp['rstep']
    new_atom = Atom(new_symbol, np.asarray(new_position))

    # append new atom to system
    atoms_prime.append(new_atom)

    # get new energy
    delta_energy = atoms_prime.get_total_energy() - atoms.get_total_energy()
    # get chemical potential
    mu = chem_potentials[new_symbol]
    # calculate acceptance
    if np.random.random((1,)) < np.exp(
            min([0, -1. * beta * delta_energy + beta * mu])):
        return atoms_prime
    else:
        return None


def del_atom(atoms, chem_potentials, beta, scatter=None):
    if len(atoms) == 1:
        return None
    atoms.center()
    # make the proposed system
    atoms_prime = dc(atoms)
    if scatter is None:
        del_atom_index = np.random.choice(range(len(atoms)))
    else:
        intensities = scatter.get_atomic_3d_overlap(atoms)
        prob = 1 - intensities
        del_atom_index = np.random.choice(range(len(atoms)), p=prob)
    del_symbol = atoms_prime[del_atom_index].symbol

    # append new atom to system
    del atoms_prime[del_atom_index]

    # get new energy
    delta_energy = atoms_prime.get_total_energy() - atoms.get_total_energy()
    # get chemical potential
    mu = chem_potentials[del_symbol]
    # calculate acceptance
    if np.random.random() < np.exp(
            min([0, -1. * beta * delta_energy - beta * mu
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
                 restart=None, logfile=None, trajectory=None, seed=None,
                 verbose=False):
        Ensemble.__init__(self, atoms, restart, logfile, trajectory, seed,
                          verbose)
        self.beta = 1. / (temperature * kB)
        self.chem_pot = chemical_potentials
        self.metadata = {'rejected_additions': 0, 'accepted_removals': 0,
                         'accepted_additions': 0, 'rejected_removals': 0}

    def step(self):
        if self.random_state.uniform() >= .5:
            mv = 'add'
            new_atoms = del_atom(self.traj[-1], self.chem_pot, self.beta)
        else:
            mv = 'remove'
            new_atoms = add_atom(self.traj[-1], self.chem_pot, self.beta)
        if new_atoms is not None:
            if self.verbose:
                print '\t' + mv + ' atom accepted'

            if mv == 'add':
                self.metadata['accepted_additions'] += 1
            elif mv == 'remove':
                self.metadata['accepted_removals'] += 1

            self.traj.append(new_atoms)
            return [new_atoms]
        else:
            if self.verbose:
                print '\t' + mv + ' atom rejected'

            if mv == 'add':
                self.metadata['rejected_additions'] += 1
            elif mv == 'remove':
                self.metadata['rejected_removals'] += 1

            return None

    def estimate_simulation_duration(self, atoms, iterations):
        t2 = time()
        e = atoms.get_potential_energy()
        te = time() - t2

        total_time = 0.
        for i in xrange(iterations):
            total_time += te
        return total_time
