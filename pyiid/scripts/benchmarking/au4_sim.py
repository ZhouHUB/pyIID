__author__ = 'christopher'
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.atoms import Atoms

from pyiid.wrappers.gpu_wrap import wrap_pdf
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.sim.dynamics import simulate_dynamics

from copy import deepcopy as dc


atomsio = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
wrap_atoms(atomsio)

qmax = 25
qbin = .1
qmax_bin = int(qmax / qbin)
n = len(atomsio)

atoms = dc(atomsio)
pdf, fq = wrap_pdf(atoms, qmin=0.0, qbin=.1, qmax=qmax)

calc = PDFCalc(gobs=pdf, qmin=0, conv=.1, qbin=.1, qmax=qmax)
atoms.set_calculator(calc)

atoms.positions *= 1.05
rwi = atoms.get_potential_energy()
print rwi
# atoms.set_momenta(np.random.normal(0, 1, (len(atoms), 3)))
atoms.set_momenta(np.zeros((len(atoms), 3)))

traj = simulate_dynamics(atoms, .5e-3, 100)

pe_list = []
chi_list = []
for a in traj:
    pe_list.append(a.get_potential_energy())
    f = a.get_forces()
    f2 = f*2

min = np.argmin(pe_list)
print 'start rw', rwi, 'end rw', pe_list[min], 'rw change', pe_list[min]-rwi
view(traj)

# '''
r = np.arange(0, 40, .01)
# r = np.arange(0, 4000, 1)
plt.plot(r[200:1200], pdf[200:1200], label='ideal')
plt.plot(r[200:1200], wrap_pdf(traj[min], qmin= 0)[0][200:1200], label='best: '+str(min)+' out of '+str(len(traj)-1))
plt.plot(r[200:1200], wrap_pdf(traj[0], qmin= 0)[0][200:1200], 'k', label='start')
plt.legend()
plt.show()
# '''