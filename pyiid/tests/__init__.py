import numpy as np
from ase.atoms import Atoms
from pyiid.wrappers.elasticscatter import wrap_atoms
from numpy.testing import assert_allclose
from itertools import *
import os
from copy import deepcopy as dc
import random
from pyiid.testing.decorators import *

from pyiid.calc.spring_calc import Spring
from pyiid.sim.dynamics import classical_dynamics
from pyiid.sim.nuts_hmc import nuts

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


def stats_check(ans1, ans2, rtol=None, atol=None):
    print 'bulk statistics:'
    print 'max', np.max(np.abs(ans2 - ans1)),
    print 'min', np.min(np.abs(ans2 - ans1)),
    print 'men', np.mean(np.abs(ans2 - ans1)),
    print 'med', np.median(np.abs(ans2 - ans1)),
    print 'std', np.std(np.abs(ans2 - ans1))

    if rtol is not None and atol is not None and len(ans1) > 1:
        print 'normalized max', np.max(np.abs(ans2 - ans1)) / ans2[
        np.unravel_index(np.argmax(np.abs(ans2 - ans1)), ans2.shape)]
        fails = np.where(np.abs(ans1 - ans2) >= atol + rtol * np.abs(ans2))
        print
        print 'allclose failures'
        print zip(ans1[fails].tolist(), ans2[fails].tolist())
        print
        print 'allclose internals'
        print zip(np.abs(ans1[fails] - ans2[fails]).tolist(),
                  (atol + rtol * np.abs(ans2[fails])).tolist())


# Setup lists of test variables
test_exp = [None]
test_atom_squares = [setup_atomic_square()]
test_potentials = [
    ('rw', .9),
    ('chi_sq', 10)
]
test_qbin = [.1]
test_spring_kwargs = [{'k': 100, 'rt': 5., 'sp_type': 'rep'},
                      {'k': 100, 'rt': 1., 'sp_type': 'com'},
                      {'k': 100, 'rt': 1., 'sp_type': 'att'}]

test_calcs = [Spring(**t_kwargs) for t_kwargs in test_spring_kwargs]

# Travis CI has certain restrictions on memory and GPU availability so we
# change the size of the tests to run
travis = False
if os.getenv('TRAVIS') or True:
    travis = True
    # use a smaller test size otherwise travis stalls
    ns = [10, 100]
    test_exp.extend([generate_experiment() for i in range(3)])
    test_atoms = [setup_atoms(int(n)) for n in ns]
    test_double_atoms = [setup_double_atoms(int(n)) for n in ns]

    # Travis doesn't have GPUs so only CPU testing
    proc_alg_pairs = list(product(['CPU'], ['nxn', 'flat']))
    comparison_pro_alg_pairs = list(combinations(proc_alg_pairs, 2))
    test_calcs.extend(['PDF', 'FQ'])

else:
    ns = np.logspace(1, 3, 3)
    test_exp.extend([generate_experiment() for i in range(3)])
    test_atoms = [setup_atoms(int(n)) for n in ns]
    test_double_atoms = [setup_double_atoms(int(n)) for n in ns]
    proc_alg_pairs = [('CPU', 'flat'), ('Multi-GPU', 'flat'),
                      # ('CPU', 'nxn'),
                      ]

    # Note there is only one CPU nxn comparison test, the CPU nxn code is
    # rather slow, thus we test it against the flattened Multi core CPU code,
    # which is much faster.  Then we run all tests agains the CPU flat kernels.
    # Thus it is imperative that the flat CPU runs with no errors.

    comparison_pro_alg_pairs = [(('CPU', 'flat'), ('Multi-GPU', 'flat'))
                                # (('CPU', 'nxn'), ('CPU', 'flat')),
                                ]
    test_calcs.extend(['PDF', 'FQ'])
