__author__ = 'christopher'
import numpy as np
from ase.atoms import Atoms

from pyiid.wrappers.scatter import Scatter
from pyiid.tests import generate_experiment
from pyiid.testing.decorators import known_fail_if
from pyiid.calc.oo_pdfcalc import PDFCalc


def setup_atomic_configs():
    atoms1 = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])
    atoms2 = atoms1.copy()
    atoms2.positions *= .75
    return atoms1, atoms2


def test_rw():
    atoms1, atoms2 = setup_atomic_configs()
    scat = Scatter()
    gobs = scat.get_pdf(atoms1)
    calc = PDFCalc(gobs=gobs, scatter=scat, potential='rw')
    atoms2.set_calculator(calc)