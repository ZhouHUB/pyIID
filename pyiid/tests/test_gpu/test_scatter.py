__author__ = 'christopher'
from numpy.testing import assert_allclose
from pyiid.wrappers.scatter import Scatter

from pyiid.tests import generate_experiment, setup_atoms
from ase.atoms import Atoms
import numpy as np


def test_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n))
            scat.set_processor('Multi-GPU')
            gpu = scat.fq(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.fq(atoms)

            assert_allclose(gpu, cpu)


def test_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            scat.set_processor('Multi-GPU')
            gpu = scat.get_pdf(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_pdf(atoms)

            assert_allclose(gpu, cpu)


def test_grad_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n))
            scat.set_processor('Multi-GPU')
            gpu = scat.get_grad_fq(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_grad_fq(atoms)
            assert_allclose(gpu, cpu)


def test_grad_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n))
            scat.set_processor('Multi-GPU')
            gpu = scat.get_grad_pdf(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_grad_pdf(atoms)
            assert_allclose(gpu, cpu)

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)