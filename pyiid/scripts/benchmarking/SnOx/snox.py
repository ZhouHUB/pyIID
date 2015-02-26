__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.sim.hmc import run_hmc
from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.utils import tag_surface_atoms, build_sphere_np, load_gr_file

# Load experimental Data
r, gr = load_gr_file('SnO2_300K-sum_00608_637_sqmsklargemsk.gr')
r = r[:-1]
gr = gr[:-1]

# Build NP
unit = aseio.read('1000062.cif')
atomsio = build_sphere_np('1000062.cif', 23./2)

wrap_atoms(atomsio)

# Experimental Parameters
qmax = 25
qbin = .1
qmin = 0.666667
qmax_bin = int(qmax / qbin)

n = len(atomsio)
atoms = dc(atomsio)
atoms.rattle(.15)

# Starting PDF
pdf, fq = wrap_pdf(atoms, qmin=qmin, qbin=qbin)
scale = np.dot(np.dot(1. / (np.dot(pdf.T, pdf)), pdf.T), gr)
print 1/scale
gr /= scale
'''
plt.plot(r, gr, label='ideal')
plt.plot(r,pdf, label='start')
plt.legend()
plt.show()
'''
calc = PDFCalc(gobs=gr, qmin=qmin, conv=1000, qbin=.1, potential='rw')
# calc = PDFCalc(gobs=gr, qmin=qmin, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print(rwi)
# atoms.set_momenta(np.zeros((len(atoms), 3)))
atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))

wtraj = PickleTrajectory('SnOx_rw_rattle_15.traj', 'w')
pe_list = []
traj, accept_list = run_hmc(atoms, 100, .8e-3, 150, 0.9, 0, .9,
                                       1.02, .98, .000001, .65, 1, wtraj=wtraj)

# traj, accept_list = run_hmc(atoms, 100, 1e-3, 30, 0.9, 0, .9,
#                                        1.02, .98, .000001, .65, 1)
for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
print 'start rw', rwi, 'end rw', pe_list[-1], 'rw change', pe_list[-1]-rwi
rw, scale, apdf, afq = wrap_rw(traj[-1], gr, qmin=qmin, qbin=.1)
print rw

# file_add_on = '_gpu_hmc_surf_exp_T1'

for atoms in traj:
    p = atoms.get_potential_energy()
    p2 = p*2
    f = atoms.get_forces()
    f2 = f*2
    # wtraj.write(atoms)

view(traj)
# r = np.arange(0, 40, .01)
plt.plot(r, gr, 'b-', label='ideal')
plt.plot(r, wrap_pdf(traj[0], qmin= qmin)[0], 'k-', label='start')
plt.plot(r, wrap_pdf(traj[-1], qmin= qmin)[0], 'r-', label='final')
plt.legend()
plt.xlabel('radius in angstrom')
plt.ylabel('PDF')
plt.title('SnOx, rattle .15')
plt.savefig('SnOx_pdf.png', bbox_inches='tight', transparent='True')
plt.show()