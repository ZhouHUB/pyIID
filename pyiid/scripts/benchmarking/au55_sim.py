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
from pyiid.calc.pdfcalc_gpu import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms


atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)

qmax = 25
qbin = .1
n = len(atomsio)

qmax_bin = int(qmax / qbin)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=0, qbin=.1)
# atoms.positions *= .95
# atoms.rattle(.05)
atoms[51].position += [1,0,0]
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=0, qbin=.1)

calc = PDFCalc(gobs=pdf, qmin=0, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print(rwi)
atoms.set_momenta(np.zeros((len(atoms), 3)))
# atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))

traj = simulate_dynamics(atoms, 5e-2, 3)

pe_list = []

for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
    f = atoms.get_forces()
    f2 = f*2

min_pe = np.argmin(pe_list)
print((rwi - traj[min_pe].get_potential_energy()))

'''
wtraj = PickleTrajectory(atoms_file_no_ext+'_gpu_hmc_contract_T5'+'.traj', 'w')
for atoms in traj:
    p = atoms.get_potential_energy()
    p2 = p*2
    f = atoms.get_forces()
    f2 = f*2
    wtraj.write(atoms)
'''
'''
view(traj)
plt.plot(pdf, label='ideal')
plt.plot(wrap_pdf(traj[0], qmin= 0)[0], label='start')
plt.plot(wrap_pdf(traj[min_pe], qmin= 0)[0], label='best:' + str(min_pe))
plt.legend()
plt.show()'''