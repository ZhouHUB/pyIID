__author__ = 'christopher'
from numpy.testing import assert_allclose
import numpy as np

from pyiid.wrappers.scatter import Scatter
from pyiid.tests import setup_atoms, generate_experiment
from pyiid.calc.oo_pdfcalc import PDFCalc


def test_calc_rw():
    # Test a random set of experimental parameters
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n), exp)
            atoms2 = setup_atoms(n, exp)

            scat.set_processor('Multi-GPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
            gpu = calc.calculate_energy(atoms2)

            scat.set_processor('Serial-CPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
            cpu = calc.calculate_energy(atoms2)

            assert_allclose(gpu, cpu)


def test_calc_chi_sq():
    # Test a random set of experimental parameters
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n), exp)
            atoms2 = setup_atoms(n, exp)

            scat.set_processor('Multi-GPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
            gpu = calc.calculate_energy(atoms2)

            scat.set_processor('Serial-CPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
            cpu = calc.calculate_energy(atoms2)

            assert_allclose(gpu, cpu)


def test_calc_grad_rw():
    # Test a random set of experimental parameters
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n), exp)
            atoms2 = setup_atoms(n, exp)

            scat.set_processor('Multi-GPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
            gpu = calc.calculate_forces(atoms2)

            scat.set_processor('Serial-CPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
            cpu = calc.calculate_forces(atoms2)

            assert_allclose(gpu, cpu)


def test_calc_grad_chi_sq():
    # Test a random set of experimental parameters
    for i in range(4):
        if i == 0:
            exp = None
        else:
            exp = generate_experiment()
        scat = Scatter(exp_dict=exp)
        # Test a set of different sized ensembles
        for n in np.logspace(1, 4, 4):
            atoms = setup_atoms(int(n), exp)
            atoms2 = setup_atoms(n, exp)

            scat.set_processor('Multi-GPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
            gpu = calc.calculate_forces(atoms2)

            scat.set_processor('Serial-CPU')
            gobs = scat.get_pdf(atoms)
            calc = PDFCalc(gobs=gobs, scatter=scat, potential='chi_sq')
            cpu = calc.calculate_forces(atoms2)

            assert_allclose(gpu, cpu)


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)