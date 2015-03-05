# import matplotlib.pyplot as plt
import cProfile
cProfile.run('''
__author__ = 'christopher'
import numpy as np
import ase.io as io
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.gpu.serial_autojit_kernel import get_d_array, get_r_array, get_scatter_array, \
    get_fq_array, get_normalization_array

dpc = DebyePDFCalculator()

# Load Atoms
atoms = io.read('/home/christopher/25_nm_half.xyz')
# atoms = io.read('/home/christopher/pdfgui_np_35.xyz')

#extract data from atoms
q = atoms.get_positions()
symbols = atoms.get_chemical_symbols()

# define Q information
Qmax = 25.
# Qmin = 2.5
Qmin = 0.0
Qbin = .11846216
qmin_bin = int(Qmin / Qbin)
Qmax_bin = int(Qmax / Qbin)
Qmax_Qmin_bin_range = np.arange(qmin_bin, Qmax_bin + Qbin)
Q = np.arange(0, Qmax, Qbin)

#initialize constants
N = len(q)
print N
d = np.zeros((N, N, 3))
n_range = range(N)
range_3 = range(3)

# Get pair coordinate distance array
get_d_array(d, q, n_range)
print 'd'
print d

#Get pair distance array
r = np.zeros((N, N))
get_r_array(r, d, n_range)
print 'r'
print r

#get scatter array
scatter_array = np.zeros((N, len(Q)))
get_scatter_array(scatter_array, symbols, dpc, n_range,
                  Qmax_Qmin_bin_range, Qbin)
print 'sa'
print scatter_array
#remove self_scattering
# np.fill_diagonal(scatter_array, 0)

#get non-normalized FQ
fq = np.zeros(len(Q))
get_fq_array(fq, r, scatter_array, n_range, Qmax_Qmin_bin_range, Qbin)

#Normalize FQ
norm_array = np.zeros(len(Q))
get_normalization_array(norm_array, scatter_array, Qmax_Qmin_bin_range, n_range)
FQ = np.nan_to_num(1 / (N * norm_array) * fq)
''', sort='cumtime')
# print FQ
# plt.plot(Q, FQ)
# plt.show()