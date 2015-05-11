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
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU')
            gpu = scat.get_fq(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_fq(atoms)
            print n, np.sqrt(np.mean((cpu-gpu)**2))
            assert_allclose(gpu, cpu, rtol=5e-3)


def test_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU')
            gpu = scat.get_pdf(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_pdf(atoms)
            print n, np.sqrt(np.mean((cpu-gpu)**2))
            assert_allclose(gpu, cpu, rtol=5e-3)


def test_grad_fq():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU')
            gpu = scat.get_grad_fq(atoms)
            scat.set_processor('Serial-CPU')
            cpu = scat.get_grad_fq(atoms)
            print n, np.sqrt(np.mean((cpu-gpu)**2))
            assert_allclose(gpu, cpu, rtol=5e-3)


def test_grad_pdf():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)

            scat.set_processor('Multi-GPU')
            gpu = scat.get_grad_pdf(atoms)

            scat.set_processor('Serial-CPU')
            cpu = scat.get_grad_pdf(atoms)
            print n, np.sqrt(np.mean((cpu-gpu)**2))
            assert_allclose(gpu, cpu, rtol=5e-3)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)

    '''import matplotlib.pyplot as plt
    exp = generate_experiment()
    scat = Scatter()
    for n in np.logspace(1, 3, 3):
        print 'start', n
        atoms = setup_atoms(int(n))
        print 'gpu'
        scat.set_processor('Multi-GPU')
        gpu = scat.get_fq(atoms)
        print 'cpu'
        scat.set_processor('Serial-CPU')
        cpu = scat.get_fq(atoms)
        print np.sqrt(np.mean((cpu-gpu)**2))
        plt.plot(gpu, label='gpu')
        plt.plot(cpu, label='cpu')
        plt.show()
        plt.plot(gpu - cpu)
        plt.show()'''