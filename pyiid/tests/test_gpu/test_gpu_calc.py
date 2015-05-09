__author__ = 'christopher'
from ase.atoms import Atoms
import numpy as np

from pyiid.wrappers.scatter import Scatter


def setup_atoms(n):
    q = np.random.random((n, 3)) * 10
    atoms = Atoms('Au' + n, q)
    return atoms


def generate_experiment():
    exp_dict = {'qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep'}
    exp_ranges = [(0, 1.5), (19., 25.), (.8, .12), (0., 2.5), (30., 50.),
                  (.005, .015)]

    for n, k in enumerate(exp_dict):
        exp_dict[k] = np.random.uniform(exp_ranges[n][0], exp_ranges[n][1])

def test_calc_rw():
