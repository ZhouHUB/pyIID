__author__ = 'christopher'
import matplotlib.pyplot as plt
from pyiid.utils import convert_atoms_to_stru
import cProfile
# cProfile.run('''
__author__ = 'christopher'
import numpy as np
import ase.io as io
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.serial_kernel import get_d_array, get_r_array, get_scatter_array, \
    get_fq_array, get_normalization_array, get_pdf_at_qmin, fq_grad_position
from scipy.fftpack import idst
from copy import deepcopy as dc
from ase import Atoms
dpc = DebyePDFCalculator()

atoms = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
q = atoms.get_positions()
symbols = atoms.get_chemical_symbols()

# define Q information
Qmax = 25.
rmax = 40
# Qmin = 2.5
Qmin = 0.0
Qbin = np.pi/(rmax+6*2*np.pi/25)
qmin_bin = int(Qmin / Qbin)
Qmax_bin = int(Qmax / Qbin)
Q = np.arange(Qmin, Qmax, Qbin)
rstep = .01

#initialize constants
N = len(q)
#print N
d = np.zeros((N, N, 3))
n_range = range(N)
range_3 = range(3)

# Get pair coordinate distance array
get_d_array(d, q, N)

#print 'd'
#print d

#Get pair distance array
r = np.zeros((N, N))
get_r_array(r, d, N)
#print 'r'
#print r

#get scatter array
scatter_array = np.zeros((N, len(Q)))
get_scatter_array(scatter_array, symbols, dpc, N, qmin_bin, Qmax_bin, Qbin)
#print 'sa'
#print scatter_array
#remove self_scattering
# np.fill_diagonal(scatter_array, 0)

#get non-normalized FQ
fq = np.zeros(len(Q))
get_fq_array(fq, r, scatter_array, N, qmin_bin, Qmax_bin, Qbin)

#Normalize FQ
norm_array = np.zeros(len(Q))
get_normalization_array(norm_array, scatter_array, qmin_bin, Qmax_bin, N)
start_FQ = np.nan_to_num(1 / (N * norm_array) * fq)
a = .00001
total_FQ = np.zeros((N, 3, len(Q)))
# for tx in range(N):
#     for tz in range(3):
#         atoms2 = dc(atoms)
#         atoms2[tx].position[tz] += a
#         q = atoms2.get_positions()
#         symbols = atoms.get_chemical_symbols()
#
#         # define Q information
#         Qmax = 25.
#         rmax = 40
#         # Qmin = 2.5
#         Qmin = 0.0
#         Qbin = np.pi/(rmax+6*2*np.pi/25)
#         qmin_bin = int(Qmin / Qbin)
#         Qmax_bin = int(Qmax / Qbin)
#         Q = np.arange(Qmin, Qmax, Qbin)
#         rstep = .01
#
#         #initialize constants
#         N = len(q)
#         #print N
#         d = np.zeros((N, N, 3))
#         n_range = range(N)
#         range_3 = range(3)
#
#         # Get pair coordinate distance array
#         get_d_array(d, q, N)
#
#         #print 'd'
#         #print d
#
#         #Get pair distance array
#         r = np.zeros((N, N))
#         get_r_array(r, d, N)
#         #print 'r'
#         #print r
#
#         #get scatter array
#         scatter_array = np.zeros((N, len(Q)))
#         get_scatter_array(scatter_array, symbols, dpc, N, qmin_bin, Qmax_bin, Qbin)
#         #print 'sa'
#         #print scatter_array
#         #remove self_scattering
#         # np.fill_diagonal(scatter_array, 0)
#
#         #get non-normalized FQ
#         fq = np.zeros(len(Q))
#         get_fq_array(fq, r, scatter_array, N, qmin_bin, Qmax_bin, Qbin)
#
#         #Normalize FQ
#         norm_array = np.zeros(len(Q))
#         get_normalization_array(norm_array, scatter_array, qmin_bin, Qmax_bin, N)
#         FQ = np.nan_to_num(1 / (N * norm_array) * fq)
#         total_FQ[tx, tz, :] += FQ
# print total_FQ
# plt.plot(total_FQ[2,2,:]-start_FQ)
# plt.show()
import math
tx = 2
tz = 1
grad_p = np.zeros((N, 3, len(Q)))
for tx in range(N):
    for tz in range(3):
        for ty in range(N):
            if tx != ty:
                for kq in range(qmin_bin, Qmax_bin):
                    sub_grad_p = \
                        scatter_array[tx, kq] * \
                        scatter_array[ty, kq] * \
                        d[tx, ty, tz] * \
                        (
                         (kq*Qbin) *
                         r[tx, ty] *
                         math.cos(kq*Qbin * r[tx, ty]) -
                         math.sin(kq*Qbin * r[tx, ty])
                        )\
                        /(r[tx, ty]**3)
                    grad_p[tx, tz, kq] += sub_grad_p
        old_settings = np.seterr(all='ignore')
        grad_p[tx, tz] = np.nan_to_num(1 / (N * norm_array) * grad_p[tx, tz])
        np.seterr(**old_settings)
partial = get_pdf_at_qmin(grad_p[2, 1,:], rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)
plt.plot(
    np.arange(0, 40, .01), partial
    # Q, total_FQ[2,1,:],
    # Q, -grad_p[3, 2,:]
)
plt.show()