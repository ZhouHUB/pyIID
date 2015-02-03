__author__ = 'christopher'
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.atoms import Atoms

from pyiid.wrappers.gpu_wrap import wrap_pdf
from pyiid.calc.pdfcalc_gpu import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.sim.nuts_hmc import nuts_da_hmc


atomsio = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
wrap_atoms(atomsio)


qmax = 100
qbin = .1
qmax_bin = int(qmax / qbin)
n = len(atomsio)


atoms = dc(atomsio)

pdf, fq = wrap_pdf(atoms, qmin=2.5, qbin=.1, qmax=qmax)
# plt.plot(pdf)
pdf, fq = wrap_pdf(atoms, qmin=0, qbin=.1, qmax=qmax)
# plt.plot(pdf)
# plt.show()

# atoms.positions *= .965

calc = PDFCalc(gobs=pdf, qmin=0, conv=1, qbin=.1, qmax=qmax)
atoms.set_calculator(calc)

'''
e_list = []
traj = []
ia = np.arange(1.1, 0.9, -.001)
for i in ia:
    atoms2 = dc(atoms)
    atoms2.positions *= i
    # print(i, atoms2.get_potential_energy())
    e_list.append(atoms2.get_potential_energy())
    atoms2.get_forces()
    traj += [atoms2]

view(traj)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(ia*3.0, e_list)

ax2.plot(ia, np.zeros(len(e_list)))
plt.show()
# '''

# '''
atoms.positions *= .95
rwi = atoms.get_potential_energy()
# print rwi
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
plt.plot(r[200:1200], wrap_pdf(traj[min], qmin= 0)[0][200:1200], label='best: '+str(min)+' out of '+str(len(traj)-1))
plt.plot(r[200:1200], wrap_pdf(traj[0], qmin= 0)[0][200:1200], 'k', label='start')
plt.legend()
plt.show()
# '''