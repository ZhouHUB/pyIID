__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio
from ase.constraints import FixAtoms

from pyiid.sim.dynamics import simulate_dynamics
from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.utils import tag_surface_atoms, load_gr_file, build_sphere_np

# Load experimental Data
r, gr = load_gr_file('SnO2_300K-sum_00608_637_sqmsklargemsk.gr')
plt.plot(r, gr)

# Build NP
unit = aseio.read('1000062.cif')
atomsio = build_sphere_np('1000062.cif', 23./2)
# view(atomsio)
wrap_atoms(atomsio)

# Experimental Parameters
qmax = 25
qbin = .1
qmin = 0.666667

qmax_bin = int(qmax / qbin)

n = len(atomsio)
atoms = dc(atomsio)
# Starting PDF
pdf, fq = wrap_pdf(atoms, qmin=qmin, qbin=qbin)

'''
plt.plot(r[:-1], gr[:-1], label='ideal')
plt.plot(r[:-1], pdf*.25, label='start')
plt.legend()
plt.show()
'''

# calc = PDFCalc(gobs=pdf, qmin=0, conv=1, qbin=.1, potential='rw')
calc = PDFCalc(gobs=gr[:-1]*4, qmin=qmin, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print(rwi)
atoms.set_momenta(np.zeros((len(atoms), 3)))
# atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3))*10)
# print atoms.get_kinetic_energy()

# traj = simulate_dynamics(atoms, 1e-2, 60)
traj = simulate_dynamics(atoms, .5e-3, 60)

pe_list = []

for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
    f = atoms.get_forces()
    f2 = f * 2

min_pe = np.argmin(pe_list)
print 'start rw', rwi, 'end rw', pe_list[min_pe], 'rw change', pe_list[
                                                                   min_pe] - rwi

'''
wtraj = PickleTrajectory(atoms_file_no_ext+'_gpu_hmc_contract_T5'+'.traj', 'w')
for atoms in traj:
    p = atoms.get_potential_energy()
    p2 = p*2
    f = atoms.get_forces()
    f2 = f*2
    wtraj.write(atoms)
'''

view(traj)
r = np.arange(0, 40, .01)
plt.plot(r, pdf, label='ideal')
plt.plot(r, wrap_pdf(traj[0], qmin=qmin)[0], label='start')
plt.plot(r, wrap_pdf(traj[min_pe], qmin=qmin)[0], label='best:' + str(min_pe))
plt.legend()
plt.xlabel('r in A')
plt.show()