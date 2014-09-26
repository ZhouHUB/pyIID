__author__ = 'christopher'
import time

import ase.io as io
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
from pyiid.grad import mc_grad
from pyiid.utils import load_gr_file
from pyiid.potential_core import Debye_srreal_U
from pyiid.alg import HMC
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.utils import convert_atoms_to_stru

#load up pdf data
r, gr = load_gr_file('/home/christopher/shared/PDFdata/July/X17A/1/FinalSum_1v2.gr')
#loat up inital atomic configuration
atoms = io.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
#set current atoms
current_atoms = atoms
#set iterator
i = 0
#for timing purposes
t0 = time.clock()
#set lists for after the fact analysis
move_list =[]
u_list =[]
k_list = []
#this process is slow so only take some of the atoms
new_atoms = current_atoms[:100]

#initial image for atoms
traj =[new_atoms]
#get initial potential energy
current_U = Debye_srreal_U(new_atoms, gr)
#start
while i < 10:
    #Run the Hamiltonian engine
    new_atoms, move_type, u, k = HMC(new_atoms,  Debye_srreal_U, current_U,
                                     mc_grad, gr, 1, .1,  3,  1e-6)
    #append a bunch of stuff
    traj += [new_atoms]
    move_list.append(move_type)
    print i, u, k, move_type
    u_list.append(u)
    k_list.append(k)
    #iterate
    i += 1
#final time
t1 = time.clock()

print 'time = ', (t1-t0), ' sec'
print u_list[-1]
# print(traj[-1].get_positions())
#make pdf plots for comparison
baseline = np.min(gr)*1.1
dpc = DebyePDFCalculator()
dpc.qmax = 25
dpc.rmin = 0.00
dpc.rmax = 40.01
gcalc = dpc(convert_atoms_to_stru(traj[-1]), qmin=2.5)[1]
gdiff = gr-gcalc
plt.plot(u_list), plt.show()


plt.figure()
plt.plot(r, gr, 'bo', label="G(r) data")
plt.plot(r, gcalc, 'r-', label="G(r) fit")
plt.plot(r, gdiff + baseline, 'g-', label="G(r) diff")
plt.plot(r, np.zeros_like(r) + baseline, 'k:')
plt.xlabel(r"$r (\AA)$")
plt.ylabel(r"$G (\AA^{-2})$")
plt.legend()

plt.show()
#view the configurations
view(traj)