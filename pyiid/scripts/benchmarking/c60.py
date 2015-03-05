__author__ = 'christopher'
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
import ase.io as aseio

from pyiid.wrappers.three_d_gpu_wrap import wrap_pdf, wrap_rw
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.sim.hmc import run_hmc

from copy import deepcopy as dc
import os


atoms_file = \
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)

qmax = 100
qbin = .1
qmax_bin = int(qmax / qbin)
n = len(atomsio)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=0.0, qbin=.1, qmax=qmax)

calc = PDFCalc(gobs=pdf, qmin=0, conv=.01, qbin=.1, qmax=qmax)
atoms.set_calculator(calc)

atoms.positions *= .95
rwi = atoms.get_potential_energy()
print rwi
# atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))
atoms.set_momenta(np.zeros((len(atoms), 3)))

traj = simulate_dynamics(atoms, .5e-2, 100)

pe_list = []
chi_list = []
for a in traj:
    pe_list.append(a.get_potential_energy())
    f = a.get_forces()
    f2 = f*2

min_pe = np.argmin(pe_list)
print 'start rw', rwi, 'end rw', pe_list[min_pe], 'rw change', pe_list[min_pe]-rwi
print wrap_rw(traj[min_pe], pdf)[0]
view(traj)

r = np.arange(0, 40, .01)
# r = np.arange(0, 4000, 1)
plt.plot(r[100:1200], pdf[100:1200], label='ideal')
plt.plot(r[100:1200], wrap_pdf(traj[min_pe], qmin= 0)[0][100:1200], label='best: '+str(min_pe)+' out of '+str(len(traj)-1))
plt.plot(r[100:1200], wrap_pdf(traj[0], qmin= 0)[0][100:1200], 'k', label='start')
plt.plot(r[100:1200], pdf[100:1200] - wrap_pdf(traj[min_pe], qmin= 0)[0][100:1200] - 15, '--k', label='start-best')
plt.legend()
plt.show()
# '''