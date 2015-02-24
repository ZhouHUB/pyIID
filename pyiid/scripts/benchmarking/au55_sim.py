__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.sim.dynamics import simulate_dynamics
from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.utils import tag_surface_atoms

atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/' \
             'Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)

qmax = 25
# qmin = 2.5
qmin = 0.
qbin = .1
n = len(atomsio)

qmax_bin = int(qmax / qbin)

atoms = dc(atomsio)
# view(atoms)
atoms.rattle(.1)
'''
tag_surface_atoms(atoms)
for atom in atoms:
    if atom.tag == 1:
        atom.position *= 1.05
'''
pdf, fq = wrap_pdf(atoms, qmin=qmin, qbin=.1)
# view(atoms)
# atoms.positions *= 1.1
# atoms.rattle(.1)
# atoms[51].position += [1,0,0]

atoms = dc(atomsio)
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=qmin, qbin=.1)

calc = PDFCalc(gobs=pdf, qmin=qmin, conv=.1, qbin=.1)
# calc = PDFCalc(gobs=pdf, qmin=qmin, conv=1, qbin=.1, potential='rw')
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print(rwi)
atoms.set_momenta(np.zeros((len(atoms), 3)))
# atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3))*10)
# print atoms.get_kinetic_energy()

# traj = simulate_dynamics(atoms, 1e-2, 60)
traj = simulate_dynamics(atoms, .5e-3, 120)

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