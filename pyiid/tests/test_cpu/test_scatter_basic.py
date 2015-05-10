__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.scatter import Scatter
from pyiid.tests import setup_atoms, generate_experiment
from pyiid.testing.decorators import known_fail_if


def test_scatter_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            fq = scat.get_fq(atoms)
            # Check that Scatter gave back something
            assert fq is not None
            # Check that all the values are not zero
            assert np.any(fq)

def test_scatter_sq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            sq = scat.get_sq(atoms)
            # Check that Scatter gave back something
            assert sq is not None
            # Check that all the values are not zero
            assert np.any(sq)

def test_scatter_iq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            iq = scat.get_iq(atoms)
            # Check that Scatter gave back something
            assert iq is not None
            # Check that all the values are not zero
            assert np.any(iq)


def test_scatter_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            pdf = scat.get_pdf(atoms)
            # Check that Scatter gave back something
            assert pdf is not None
            # Check that all the values are not zero
            assert np.any(pdf)


@known_fail_if(True)
def test_gpu_scatter_fail():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        scat.set_processor('Multi-GPU')
        assert scat.processor == 'Multi-GPU'

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)