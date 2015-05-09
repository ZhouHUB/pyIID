__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.scatter import Scatter
from pyiid.tests import setup_atoms, generate_experiment
from pyiid.testing.decorators import known_fail_if


def test_scatter():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n))
            fq = scat.fq(atoms)
            assert fq is not None
            assert fq != np.zeros(fq.shape)
            pdf = scat.pdf(atoms)
            assert pdf is not None
            assert pdf != np.zeros(pdf.shape)

@known_fail_if(True)
def test_gpu_scatter_fail():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n))
            fq = scat.fq(atoms)
            assert fq is not None
            assert fq != np.zeros(fq.shape)
            pdf = scat.pdf(atoms)
            assert pdf is not None
            assert pdf != np.zeros(pdf.shape)