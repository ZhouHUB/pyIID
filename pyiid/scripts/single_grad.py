__author__ = 'christopher'

import ase.io as io
from ase.visualize import view
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from matplotlib.pyplot import plot, show
import numpy as np
from pyiid.potential_core import rw
from pyiid.grad import mc_grad
import time
from pyiid.utils import convert_atoms_to_stru, update_stru, load_gr_file
from pyiid.potential_core import Debye_srreal_U


r, gr = load_gr_file('/home/christopher/shared/PDFdata/July/X17A/1/FinalSum_1v2.gr')
atoms = io.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
# # print atoms
# stru = convert_atoms_to_stru(atoms)
# # print len(stru)
# # print len(atoms)
# # view(atoms)
# dpc = DebyePDFCalculator()
# dpc.qmax = 25
# dpc.rmin = 0.00
# dpc.rmax = 40.01
# r0, g0 = dpc(stru, qmin=2.5)
# atoms.translate(np.random.random((len(atoms), 3))*10)
# # view(atoms)
# # print atoms
# new_stru = update_stru(atoms, stru)
# # print new_stru
# r1, g1 = dpc(stru, qmin=2.5)
#
# # plot(r0, g0, 'k', r1, g1, 'r')
# # show()
# print(rw(g0, gr))


delta_qi = .01

print len(atoms)
current_U = Debye_srreal_U(atoms, gr)
print current_U
t0 = time.clock()
grad = mc_grad(atoms, gr, Debye_srreal_U, delta_qi)
t1 = time.clock()
print grad
np.savetxt('grad_test.txt', grad)

tf = t1-t0
print tf