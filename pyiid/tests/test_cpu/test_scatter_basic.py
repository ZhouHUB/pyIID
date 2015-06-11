__author__ = 'christopher'
import numpy as np
from pyiid.wrappers.elasticscatter import ElasticScatter
from pyiid.tests import setup_atoms, generate_experiment
from pyiid.testing.decorators import known_fail_if
from numba import cuda

try:
    cuda.gpus.lst
    gpus = True
except:
    gpus = False

def test_scatter_fq_defaults():
    """
    Test that F(Q) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        fq = scat.get_fq(atoms)
        # Check that Scatter gave back something
        assert fq is not None
        # Check that all the values are not zero
        assert np.any(fq)


def test_scatter_fq():
    """
    Test that F(Q) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(3):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        # Test a set of different sized ensembles
        print exp
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            fq = scat.get_fq(atoms)
            print fq.shape
            # Check that Scatter gave back something
            assert fq is not None
            # Check that all the values are not zero
            assert np.any(fq)


def test_scatter_sq_defaults():
    """
    Test that S(Q) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        sq = scat.get_sq(atoms)
        # Check that Scatter gave back something
        assert sq is not None
        # Check that all the values are not zero
        assert np.any(sq)


def test_scatter_sq():
    """
    Test that S(Q) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(3):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor('CPU')
        print exp
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            sq = scat.get_sq(atoms)
            # Check that Scatter gave back something
            assert sq is not None
            # Check that all the values are not zero
            assert np.any(sq)


def test_scatter_iq_defaults():
    """
    Test that I(Q) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        iq = scat.get_iq(atoms)
        # Check that Scatter gave back something
        assert iq is not None
        # Check that all the values are not zero
        assert np.any(iq)


def test_scatter_iq():
    """
    Test that I(Q) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(3):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor('CPU')
        print exp
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            iq = scat.get_iq(atoms)
            # Check that Scatter gave back something
            assert iq is not None
            # Check that all the values are not zero
            assert np.any(iq)


def test_scatter_pdf_defaults():
    """
    Test that G(r) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        pdf = scat.get_pdf(atoms)
        # Check that Scatter gave back something
        assert pdf is not None
        # Check that all the values are not zero
        assert np.any(pdf)


def test_scatter_pdf():
    """
    Test that G(r) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(3):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        print exp
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            pdf = scat.get_pdf(atoms)
            # Check that Scatter gave back something
            assert pdf is not None
            # Check that all the values are not zero
            assert np.any(pdf)


def test_scatter_grad_fq_defaults():
    """
    Test that grad F(Q) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        grad_fq = scat.get_grad_fq(atoms)
        # Check that Scatter gave back something
        assert grad_fq is not None
        # Check that all the values are not zero
        assert np.any(grad_fq)


def test_scatter_grad_fq():
    """
    Test that grad F(Q) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(3):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor('CPU')
        print exp
        # Test a set of different sized ensembles
        for n in np.logspace(1, 3, 3):
            atoms = setup_atoms(int(n), exp)
            grad_fq = scat.get_grad_fq(atoms)
            # Check that Scatter gave back something
            assert grad_fq is not None
            # Check that all the values are not zero
            assert np.any(grad_fq)


def test_scatter_grad_pdf_defaults():
    """
    Test that grad G(r) from scatter gives back non-zero numbers for default
    experimental scattering values, for the CPU implementation
    """
    exp = None
    scat = ElasticScatter(exp_dict=exp)
    scat.set_processor('CPU')
    # Test a set of different sized ensembles
    for n in np.logspace(1, 3, 3):
        atoms = setup_atoms(int(n), exp)
        grad_pdf = scat.get_grad_pdf(atoms)
        # Check that Scatter gave back something
        assert grad_pdf is not None
        # Check that all the values are not zero
        assert np.any(grad_pdf)


def test_scatter_grad_pdf():
    """
    Test that grad G(r) from scatter gives back non-zero numbers for random
    experimental scattering values, for the CPU implementation
    """
    for i in range(2):
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor('CPU')
        print exp
        # Test a set of different sized ensembles
        for n in np.logspace(1, 2, 2):
            atoms = setup_atoms(int(n), exp)
            grad_pdf = scat.get_grad_pdf(atoms)
            # Check that Scatter gave back something
            assert grad_pdf is not None
            # Check that all the values are not zero
            assert np.any(grad_pdf)

@known_fail_if(not gpus)
def test_gpu_scatter_fail():
        exp = generate_experiment()
        scat = ElasticScatter(exp_dict=exp)
        scat.set_processor('Multi-GPU')
        assert scat.processor == 'Multi-GPU'

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)