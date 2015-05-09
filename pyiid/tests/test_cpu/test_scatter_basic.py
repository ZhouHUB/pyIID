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
            assert fq is not None
            assert np.all(fq)


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
            assert pdf is not None
            assert np.all(pdf)


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