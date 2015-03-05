__author__ = 'christopher'
import cProfile
cProfile.run('''
import numpy as np
import os
import ase.io as aseio

from pyiid.sim.hmc import run_hmc
from pyiid.test_wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc_gpu import PDFCalc


atoms_file = \
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atoms = aseio.read(atoms_file)
scatter_array = np.loadtxt(
    '/mnt/bulk-data/Dropbox/'
    'BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt',
    dtype=np.float32)
atoms.set_array('scatter', scatter_array)

atoms *= (7, 1, 1)

qmax = 25
qbin = .1
n = len(atoms)

qmax_bin = int(qmax / qbin)

pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1)
atoms.positions *= 1.001
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=2.5, qbin=.1)

calc = PDFCalc(gobs=pdf, qmin=2.5, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
atoms.set_velocities(np.zeros((len(atoms), 3)))

traj, accept_list = run_hmc(atoms, 2, .1, 10, 0.9, 0, .9,
                            1.02, .98, .001, .65)
''', sort='tottime')