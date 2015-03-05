__author__ = 'christopher'
from ase.atoms import Atoms
import ase.io as aseio

from pyiid.wrappers.kernel_wrap import wrap_atoms
from pyiid.calc.pdfcalc import PDFCalc
from pyiid.wrappers.three_d_gpu_wrap import wrap_rw, wrap_pdf

import matplotlib.pyplot as plt
from pprint import pprint
import time
from copy import deepcopy as dc
from collections import OrderedDict

atoms = Atoms('Au4', [[0,0,0], [3,0,0], [0,3,0], [3,3,0]])

wrap_atoms(atoms)
atoms2 = dc(atoms)
pdf, fq = wrap_pdf(atoms, qmin=0.0, qbin=.1)


atoms.positions *= .95

# Time for Rw
gpu_3_d = []
gpu_2_d = []
cpu = []

pu = [
    'gpu_3_d',
    'gpu_2_d',
    'cpu'
]
super_results_d = OrderedDict()
for i in range(1, 150, 10):
    atoms = dc(atoms2)
    atoms *= (i, 1, 1)
    atoms.rattle()
    print i, len(atoms)
    results_d = {}
    for processor in pu:
        sub_d = {}
        calc = PDFCalc(gobs=pdf, qmin=0.0, conv=1, qbin=.1, processor=processor)
        atoms.set_calculator(calc)
        s = time.time()
        nrg = atoms.get_potential_energy()
        f = time.time()
        sub_d['time for energy'] = f-s
        sub_d['energy'] = nrg
        s = time.time()
        force = atoms.get_forces()
        f = time.time()
        sub_d['time for force'] = f-s
        # sub_d['force'] = force
        results_d[processor] = sub_d
        atoms._del_calculator
    super_results_d[i*4] = results_d
# pprint(super_results_d)

sizes = []
cpu_f = []
cpu_e = []
gpu3_d_f = []
gpu3_d_e = []
gpu2_d_f = []
gpu2_d_e = []

for key, value in super_results_d.iteritems():
    sizes.append(key)
    cpu_f.append(value['cpu']['time for force'])
    cpu_e.append(value['cpu']['time for energy'])

    gpu3_d_e.append(value['gpu_3_d']['time for energy'])
    gpu3_d_f.append(value['gpu_3_d']['time for force'])

    gpu2_d_f.append(value['gpu_2_d']['time for force'])
    gpu2_d_e.append(value['gpu_2_d']['time for energy'])

# print sizes, gpu3_d_e

plt.plot(sizes, cpu_f, 'bo', label='cpu energy')
plt.plot(sizes, cpu_e, 'b-', label='cpu force')

plt.plot(sizes, gpu3_d_e, 'ro', label='3D grid energy')
plt.plot(sizes, gpu3_d_f, 'r-', label='3D grid force')

plt.plot(sizes, gpu2_d_e, 'ko', label='2D grid energy')
plt.plot(sizes, gpu2_d_f, 'k-', label='2D grid force')
plt.legend(loc=2)
plt.xlabel('Number of atoms')
plt.ylabel('time (s) [lower is better]')
plt.title('Scaling of algorithm')
plt.show()