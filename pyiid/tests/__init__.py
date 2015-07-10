from ase import Atoms

__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms
from pyiid.wrappers.elasticscatter import wrap_atoms
import unittest
from ddt import ddt, data, unpack
from numpy.testing import assert_allclose
from itertools import *
import os

TC = unittest.TestCase


def setup_atoms(n, exp_dict=None):
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)
    wrap_atoms(atoms, exp_dict)
    return atoms


def setup_double_atoms(n, exp_dict=None):
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)
    wrap_atoms(atoms, exp_dict)

    q2 = np.random.random((n, 3)) * 10
    atoms2 = Atoms('Au' + str(int(n)), q2)
    wrap_atoms(atoms2, exp_dict)
    return atoms, atoms2


def generate_experiment():
    exp_dict = {}
    exp_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep']
    exp_ranges = [(0, 1.5), (19., 25.), (.8, .12), (0., 2.5), (30., 50.),
                  (.005, .015)]
    for n, k in enumerate(exp_keys):
        exp_dict[k] = np.random.uniform(exp_ranges[n][0], exp_ranges[n][1])
    # exp_dict['qbin'] = np.pi / (exp_dict['rmax'] + 6 * 2 * np.pi /
    #                             exp_dict['qmax'])
    return exp_dict


def setup_atomic_square():
    atoms1 = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    atoms2 = atoms1.copy()
    scale = .75
    atoms2.positions *= scale
    return atoms1, atoms2


test_exp = [None]
test_atom_squares = [setup_atomic_square()]
test_potentials = [('rw', .9), ('chi_sq', 250)]

if os.getenv('TRAVIS') or True:
    # use a smaller test size otherwise travis stalls
    ns = [100]
    test_atoms = [setup_atoms(int(n)) for n in ns]
    test_double_atoms = [setup_double_atoms(int(n)) for n in ns]
    proc_alg_pairs = list(product(['CPU'], ['nxn', 'flat']))
    comparison_pro_alg_pairs = list(combinations(proc_alg_pairs, 2))



else:
    test_exp.extend([generate_experiment() for i in range(1)])
    test_atoms = [setup_atoms(int(n)) for n in np.logspace(1, 3, 3)]
    test_double_atoms = [setup_double_atoms(int(n)) for n in
                         np.logspace(1, 3, 3)]
    # test_double_atoms = [setup_double_atoms(n) for n in np.logspace(1, 1, 1)]
    proc_alg_pairs = list(product(['CPU', 'Multi-GPU'], ['nxn', 'flat']))

    # Note there is only one CPU nxn comparison test, the CPU nxn code is
    # rather slow, thus we test it against the flattened Multi core CPU code,
    # which is much faster.  Then we run all tests agains the CPU flat kernels.
    # Thus it is imperative that the flat CPU runs with no errors.
    comparison_pro_alg_pairs = [(('CPU', 'nxn'), ('CPU', 'flat'))]
    comparison_pro_alg_pairs.extend(
        list(combinations(proc_alg_pairs[1:], 2))[:-1])

test_qbin = [.1]
