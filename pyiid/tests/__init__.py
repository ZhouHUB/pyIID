from ase import Atoms
import numpy as np
from ase.atoms import Atoms
from pyiid.wrappers.elasticscatter import wrap_atoms
import unittest
from numpy.testing import assert_allclose
from itertools import *
import os
from copy import deepcopy as dc
import random
from pyiid.testing.decorators import *

__author__ = 'christopher'


def setup_atoms(n):
    """
    Generate a configuration of n gold atoms with random positions
    """
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)
    return atoms


def setup_double_atoms(n):
    """
    Generate two configuration of n gold atoms with random positions
    """
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + str(int(n)), q)

    q2 = np.random.random((n, 3)) * 10
    atoms2 = Atoms('Au' + str(int(n)), q2)
    return atoms, atoms2


def generate_experiment():
    """
    Generate elastic scattering experiments which are reasonable but random
    """
    exp_dict = {}
    exp_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep']
    exp_ranges = [(0, 1.5), (19., 25.), (.8, .12), (0., 2.5), (30., 50.),
                  (.005, .015)]
    for n, k in enumerate(exp_keys):
        exp_dict[k] = np.random.uniform(exp_ranges[n][0], exp_ranges[n][1])
    exp_dict['sampling'] = random.choice(['full', 'ns'])
    return exp_dict


def setup_atomic_square():
    """
    Setup squares of 4 gold atoms with known positions
    :return:
    """
    atoms1 = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    atoms2 = atoms1.copy()
    scale = .75
    atoms2.positions *= scale
    return atoms1, atoms2

def stats_check(ans1, ans2):
    return np.max(np.abs(ans2 - ans1)), np.mean(
        np.abs(ans2 - ans1)), np.std(np.abs(ans2 - ans1))

# Setup lists of test variables
test_exp = [None]
test_atom_squares = [setup_atomic_square()]
test_potentials = [
    ('rw', .9),
    ('chi_sq', 10)
]
test_qbin = [.1]

# Travis CI has certain restrictions on memory and GPU availability so we
# change the size of the tests to run
travis = False
if os.getenv('TRAVIS'):
    # if os.getenv('TRAVIS') or True:
    travis = True

    # use a smaller test size otherwise travis stalls
    ns = [10, 100]
    test_exp.extend([generate_experiment() for i in range(3)])
    test_atoms = [setup_atoms(int(n)) for n in ns]
    test_double_atoms = [setup_double_atoms(int(n)) for n in ns]

    # Travis doesn't have GPUs so only CPU testing
    proc_alg_pairs = list(product(['CPU'], ['nxn', 'flat']))

    # proc_alg_pairs = [('CPU', 'flat'), ('Multi-GPU', 'flat')]
    comparison_pro_alg_pairs = list(combinations(proc_alg_pairs, 2))

else:
    ns = np.logspace(1, 3, 3)
    test_exp.extend([generate_experiment() for i in range(3)])
    test_atoms = [setup_atoms(int(n)) for n in ns]
    test_double_atoms = [setup_double_atoms(int(n)) for n in ns]
    # proc_alg_pairs = list(product(['CPU', 'Multi-GPU'], ['nxn', 'flat']))
    proc_alg_pairs = [
        # ('CPU', 'nxn'),
        ('CPU', 'flat'), ('Multi-GPU', 'flat')]

    # Note there is only one CPU nxn comparison test, the CPU nxn code is
    # rather slow, thus we test it against the flattened Multi core CPU code,
    # which is much faster.  Then we run all tests agains the CPU flat kernels.
    # Thus it is imperative that the flat CPU runs with no errors.

    comparison_pro_alg_pairs = [
        # (('CPU', 'nxn'), ('CPU', 'flat')),
        (('CPU', 'flat'), ('Multi-GPU', 'flat'))
    ]
    # comparison_pro_alg_pairs = []
    # comparison_pro_alg_pairs.extend(
    #     list(combinations(proc_alg_pairs[1:], 2))[:-1])
    # comparison_pro_alg_pairs = [(('CPU', 'flat'), ('Multi-GPU', 'flat'))]
