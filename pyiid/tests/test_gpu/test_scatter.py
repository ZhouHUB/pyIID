__author__ = 'christopher'
from numpy.testing import assert_allclose
from pyiid.wrappers.elasticscatter import ElasticScatter

from pyiid.tests import generate_experiment, setup_atoms
from ase.atoms import Atoms
import numpy as np
import matplotlib.pyplot as plt

rt = 1e-5
at = .000005 

def test_fq_flat():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'flat')
            gpu = scat.get_fq(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_fq(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'
'''

def test_fq_nxn():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'nxn')
            gpu = scat.get_fq(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_fq(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'
'''


def test_pdf_flat():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'flat')
            gpu = scat.get_pdf(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_pdf(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'

'''
def test_pdf_nxn():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'nxn')
            gpu = scat.get_pdf(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_pdf(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'


def test_grad_fq_flat():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'flat')
            gpu = scat.get_grad_fq(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_grad_fq(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'


def test_grad_fq_nxn():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            scat.set_processor('Multi-GPU', 'nxn')
            gpu = scat.get_grad_fq(atoms)
            scat.set_processor('CPU')
            cpu = scat.get_grad_fq(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'


def test_grad_pdf_flat():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)

            scat.set_processor('Multi-GPU', 'flat')
            gpu = scat.get_grad_pdf(atoms)

            scat.set_processor('CPU')
            cpu = scat.get_grad_pdf(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'


def test_grad_pdf_nxn():
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)

            scat.set_processor('Multi-GPU', 'nxn')
            gpu = scat.get_grad_pdf(atoms)

            scat.set_processor('CPU')
            cpu = scat.get_grad_pdf(atoms)
            # print '\n', n
            # print 'rms', np.sqrt(np.mean((gpu - cpu) ** 2))
            # print 'mean difference', np.mean(gpu - cpu)
            # print 'median difference', np.median(gpu - cpu)
            assert_allclose(gpu, cpu, rtol=rt, atol=at)
            # print 'passed'

'''
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
