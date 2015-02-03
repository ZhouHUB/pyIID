__author__ = 'christopher'
import numpy as np
import matplotlib.pyplot as plt

import os
from copy import deepcopy as dc

from ase.visualize import view
from ase.io.trajectory import PickleTrajectory
import ase.io as aseio

from pyiid.hmc import run_hmc
from pyiid.wrappers.gpu_wrap import wrap_rw, wrap_pdf
from pyiid.pdfcalc_gpu import PDFCalc
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
pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1)
atoms.positions *= .95
# atoms.rattle(.1)
rw, scale, apdf, afq = wrap_rw(atoms, pdf, qmin=2.5, qbin=.1)

calc = PDFCalc(gobs=pdf, qmin=2.5, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
atoms.set_velocities(np.zeros((len(atoms), 3)))

pe_list = []
traj, accept_list, move_list = run_hmc(atoms, 30, .1, 12, 0.9, 0, .9,
                                       1.02, .98, .001, .65, 5)
for atoms in traj:
    pe_list.append(atoms.get_potential_energy())
print((rwi - traj[-1].get_potential_energy()))

wtraj = PickleTrajectory(atoms_file_no_ext+'_gpu_hmc_contract_T5'+'.traj', 'w')
for atoms in traj:
    p = atoms.get_potential_energy()
    p2 = p*2
    f = atoms.get_forces()
    f2 = f*2
    wtraj.write(atoms)

view(traj)
plt.plot(pdf, label='ideal')
plt.plot(wrap_pdf(traj[0], qmin= 2.5)[0], label='start')
plt.plot(wrap_pdf(traj[-1], qmin= 2.5)[0], label='final')
plt.legend()
plt.show()