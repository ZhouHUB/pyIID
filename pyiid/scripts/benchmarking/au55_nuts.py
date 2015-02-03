__author__ = 'christopher'
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
import ase.io as aseio

from pyiid.wrappers.gpu_wrap import wrap_pdf
from pyiid.calc.pdfcalc_gpu import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.nuts_hmc import nuts_da_hmc


atoms_file = '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.xyz'
atoms_file_no_ext = os.path.splitext(atoms_file)[0]
atomsio = aseio.read(atoms_file)
wrap_atoms(atomsio)

qmax = 25
qbin = .1
qmax_bin = int(qmax / qbin)
n = len(atomsio)


atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=0, qbin=.1)

atoms.positions *= .975

calc = PDFCalc(gobs=pdf, qmin=0, conv=1, qbin=.1)
atoms.set_calculator(calc)
rwi = atoms.get_potential_energy()
print rwi
atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))
# atoms.set_momenta(np.zeros((len(atoms), 3)))

traj = nuts_da_hmc(atoms, .65, 1000, 10)

pe_list = []
chi_list = []
for a in traj:
    pe_list.append(a.get_potential_energy())
    f = a.get_forces()
    f2 = f*2

min = np.argmin(pe_list)
print 'start rw', rwi, 'end rw', pe_list[min], 'rw change', pe_list[min]-rwi
view(traj)

r = np.arange(0, 40, .01)
# r = np.arange(0, 4000, 1)
plt.plot(r[200:1200], pdf[200:1200], label='ideal')
# for num, a in enumerate(traj):
#     plt.plot(wrap_pdf(a, qmin= 2.5)[0], label=str(num))
plt.plot(r[200:1200], wrap_pdf(traj[min], qmin= 2.5)[0][200:1200], label='best: '+str(min)+' out of '+str(len(traj)))
plt.plot(r[200:1200], wrap_pdf(traj[0], qmin= 2.5)[0][200:1200], 'k', label='start')
plt.legend()
plt.show()