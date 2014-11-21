import cProfile
import matplotlib.pyplot as plt
# cProfile.run('''
__author__ = 'christopher'
import numpy as np
import ase.io as io
import math
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.utils import convert_atoms_to_stru, convert_stru_to_atoms
from pyiid.serial_kernel import get_d_array, get_r_array, get_scatter_array, get_fq_array, get_normalization_array
dpc = DebyePDFCalculator()

# import time
# t0 = time.time()

#Load Atoms
# atoms = io.read('/home/christopher/25_nm_half.xyz')
atoms = io.read('/home/christopher/pdfgui_np_35.xyz')
q = atoms.get_positions()
symbols = atoms.get_chemical_symbols()
#define Q information

Qmax = 25.
Qmin = 2.5
Qbin = .11846216
Qmin_bin = int(Qmin/Qbin)
Qmax_bin = int(Qmax/Qbin)
Qmax_Qmin_bin_range = np.arange(Qmin_bin, Qmax_bin+Qbin)
Q = np.arange(0, Qmax, Qbin)

# print q
N = len(q)
print N
d = np.zeros((N, N, 3))
n_range = range(N)
range_3 = range(3)
#Get pair coordinate distance array
get_d_array(d, q, n_range, range_3)
print d

# t1 = time.time()
# print d

#Get pair distance array
r = np.zeros((N, N))
get_r_array(r, d, n_range)
print r
#get scatter array
scatter_array = np.zeros((N, len(Q)))
get_scatter_array(scatter_array, symbols, dpc, n_range,
                      Qmax_Qmin_bin_range, Qbin)

np.fill_diagonal(scatter_array, 0)
fq = np.zeros(len(Q))
get_fq_array(fq, r, scatter_array, n_range, Qmax_Qmin_bin_range, Qbin)

norm_array = np.zeros(len(Q))
get_normalization_array(norm_array, scatter_array, Qmax_Qmin_bin_range, n_range)
FQ = np.nan_to_num(1/(N*norm_array)*fq)
print FQ
plt.plot(Q, FQ)
plt.show()
AAA

# t2 = time.time()
# print reduced_d
rmax = 40
rbin = .01
rbins = rmax/rbin
print rbins
gr = np.zeros((rbins))
for tx in n_range:
    for ty in n_range:
        gr[int(r[tx, ty]/rbin)]+=1
# print gr
# import matplotlib.pyplot as plt
# plt.plot(np.arange(0, rmax, rbin), gr)
# plt.show()

# print t1-t0
# print t2-t1
# print t2-t0