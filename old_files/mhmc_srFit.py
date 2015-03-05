"""There is a problem with this code, it seems to copy tons of atoms without
remvoing them"""

__author__ = 'christopher'
import time
import pickle

import ase.io as io
from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt
from diffpy.srfit.pdf import PDFContribution
from diffpy.srfit.fitbase import FitRecipe

from pyiid.old_hmc.potential_core import Debye_srfit_U
from old_files.old_hmc.alg import srFit_mhmc
from pyiid.utils import convert_atoms_to_stru, convert_stru_to_atoms


atoms = io.read('/home/christopher/pdfgui_np_25_rattle1_cut.xyz')
current_atoms = atoms[:]
dataFile = '/home/christopher/7_7_7_FinalSum.gr'
nipdPDF = PDFContribution("NiPd")

nipdPDF.loadData(dataFile)
nipdPDF.setCalculationRange(xmin=1, xmax=25, dx=0.01)

nipdStructure = convert_atoms_to_stru(current_atoms)
nipdPDF.addStructure("NiPd", nipdStructure, periodic=False)

nipdFit = FitRecipe()

nipdFit.addContribution(nipdPDF)

nipdFit.addVar(nipdPDF.scale, 1)
nipdFit.addVar(nipdPDF.NiPd.delta2, 5)

nipdFit.addVar(nipdPDF.qdamp, 0.045, fixed=True)

nipdPDF.NiPd.setQmin(1.0)

nipdPDF.NiPd.setQmax(20.0)

NiBiso = nipdFit.newVar("Ni_Biso", value=1.0)
PdBiso = nipdFit.newVar("Pd_Biso", value=1.0)

atoms = nipdPDF.NiPd.phase.getScatterers()
for atom in atoms:
    if atom.element == 'Ni':
        nipdFit.constrain(atom.Biso, NiBiso)
    elif atom.element == 'Pd':
        nipdFit.constrain(atom.Biso, PdBiso)

zoomscale = nipdFit.newVar('zoomscale', value=1.0)

lattice = nipdPDF.NiPd.phase.getLattice()
nipdFit.constrain(lattice.a, zoomscale)
nipdFit.constrain(lattice.b, zoomscale)
nipdFit.constrain(lattice.c, zoomscale)

# Turn off printout of iteration number.


# print nipdFit._contributions['NiPd'].NiPd.phase.stru.xyz
# new = copy.deepcopy(nipdFit)
# new._contributions['NiPd'].NiPd.phase.stru.xyz += np.ones(
#     nipdFit._contributions['NiPd'].NiPd.phase.stru.xyz.shape)
# print nipdFit._contributions['NiPd'].NiPd.phase.stru.xyz
# nipdPDF.NiPd.phase.stru.xyz



t0 = time.clock()

traj = convert_stru_to_atoms(nipdFit._contributions[
                                       'NiPd'].NiPd.phase.stru)


move_list = []

basename = '/home/christopher/dev/IID_data/results' \
           '/srfit_mhmc_25nm_NiPd'
print 'Start current_U'
current_U = Debye_srfit_U(nipdFit)
nipdFit.clearFitHooks()
print current_U
u_list = [current_U]
iterat = 3000
start_T = .9
alpha = 10**-2.5
final_T = .1
i = 0
fixed = None
atom_len = len(current_atoms)
for i in range(iterat):
    if current_U < .20:
        break
    else:
        try:
            # T = .5 - float(i)/iterat*.99999*.5
            T = 0
            # T = start_T*np.exp(-i*alpha) + final_T
            new_fit, move_type, current_U = srFit_mhmc(atom_len, current_U,
                                                       nipdFit, T, .01,
                                                       U = Debye_srfit_U)

            new_atoms = convert_stru_to_atoms(nipdFit._contributions[
                                       'NiPd'].NiPd.phase.stru)
            traj += new_atoms
            move_list.append(move_type)
            u_list.append(current_U)
            print i, current_U, move_type, T
            i += 1
        # except:
        #     raise
        except KeyboardInterrupt:
            io.write(basename+'.traj', traj)
            with open(basename+'.txt', 'wb') as f:
                pickle.dump(u_list, f)
            with open(basename+'_bool.txt', 'wb') as f:
                pickle.dump(move_list, f)
            raise
t1 = time.clock()

print 'time = ', (t1 - t0), ' sec'
print u_list[-1]
dist_from_crystal = []
q_crystal = current_atoms.get_positions()

# Get the experimental data from the recipe
r = nipdFit.NiPd.profile.x
gobs = nipdFit.NiPd.profile.y

# Get the calculated PDF and compute the difference between the calculated and
# measured PDF
gcalc = nipdFit.NiPd.evaluate()
bapdline = 1.1 * gobs.min()
gdiff = gobs - gcalc

# Plot!
plt.figure()
plt.plot(r, gobs, 'bo', label="G(r) data")
plt.plot(r, gcalc, 'r-', label="G(r) fit")
plt.plot(r, gdiff + bapdline, 'g-', label="G(r) diff")
plt.plot(r, np.zeros_like(r) + bapdline, 'k:')
plt.xlabel(r"$r (\AA)$")
plt.ylabel(r"$G (\AA^{-2})$")
plt.legend()

plt.show()

view(traj)
io.write(basename+'.traj', traj)

with open(basename+'.txt', 'wb') as f:
    pickle.dump(u_list, f)
with open(basename+'_bool.txt', 'wb') as f:
    pickle.dump(move_list, f)