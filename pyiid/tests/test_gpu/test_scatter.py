__author__ = 'christopher'
from numpy.testing import assert_allclose
from pyiid.wrappers.scatter import Scatter
from ase.atoms import Atoms
import numpy as np


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
    return exp_dict


def test_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(n)
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
            atoms = setup_atoms(n)
            scat.set_processor('Multi-GPU')
            gpu = scat.pdf(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.pdf(atoms)

            assert_allclose(gpu, cpu)


def test_grad_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(n)
            scat.set_processor('Multi-GPU')
            gpu = scat.grad_fq(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.grad_fq(atoms)
            assert_allclose(gpu, cpu)


def test_grad_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(n)
            scat.set_processor('Multi-GPU')
            gpu = scat.grad_pdf(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.grad_pdf(atoms)
            assert_allclose(gpu, cpu)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)