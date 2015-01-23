__author__ = 'christopher'

import cProfile
# cProfile.run('''
from ase.atoms import Atoms as atoms
import ase.io as aseio
from ase.visualize import view
import matplotlib.pyplot as plt
from timeit import default_timer as time
import numpy as np
from numbapro import cuda
from pyiid.gpu.numbapro_cuda_kernels import *
from copy import deepcopy as dc

atomsio = aseio.read(
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz')
atoms = dc(atomsio)
for i in range(10):
    atoms += atomsio

q = atoms.get_positions()
qmin = .25
qmax = 25
qbin = .1

q = q.astype(np.float32)
n = len(q)
qmin_bin = 0
qmax_bin = int(qmax / qbin)

# Atoms definition, outside of calc

# scatter_array = np.zeros((n, qmax_bin))
# scatter_array = np.ones((n, qmax_bin), dtype=np.float32) * 2
# get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
scatter_array = np.loadtxt('mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt')
atoms.set_array('scatter', scatter_array)

d = np.zeros((n, n, 3), dtype=np.float32)
r = np.zeros((n, n), dtype=np.float32)
norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
tzr = np.zeros((n, n, qmax_bin), dtype=np.float32)

# scatter_array = np.zeros((n, qmax_bin))
scatter_array = np.ones((n, qmax_bin), dtype=np.float32) * 2
# get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
atoms.set_array('scatter', scatter_array)

super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)

stream = cuda.stream()
tpb = 32
bpg = int(math.ceil(float(n) / tpb))
s = time()
#push empty d, full q, and number n to GPU
dd = cuda.to_device(d, stream)
dq = cuda.to_device(q, stream)
dr = cuda.to_device(r, stream)
dfq = cuda.to_device(super_fq, stream)
dscat = cuda.to_device(scatter_array, stream)
dtzr = cuda.to_device(tzr, stream)
# dqbin = cuda.to_device(np.asarray(qbin, dtype=np.float32))
dnorm = cuda.to_device(norm_array)

get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
cuda.synchronize()

get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)
cuda.synchronize()

get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dtzr, dr, qbin)
get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)
cuda.synchronize()

get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq, dtzr)
cuda.synchronize()

get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
cuda.synchronize()

get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
cuda.synchronize()

rgrad = np.zeros((n, n, 3), dtype=np.float32)
q_over_r = np.zeros((n, n, qmax_bin), dtype=np.float32)
cos_term = np.zeros((n, n, qmax_bin), dtype=np.float32)
f_over_r = np.zeros((n, n, qmax_bin), dtype=np.float32)
grad_p = np.zeros((n, n, 3, qmax_bin), dtype=np.float32)

drgrad = cuda.to_device(rgrad, stream)
dq_over_r = cuda.to_device(q_over_r, stream)
dcos_term = cuda.to_device(cos_term, stream)
df_over_r = cuda.to_device(f_over_r, stream)
dgrad_p = cuda.to_device(grad_p, stream)


fq_grad_position0[(bpg, bpg), (tpb, tpb), stream](drgrad, dd, dr)

fq_grad_position1[(bpg, bpg), (tpb, tpb), stream](dq_over_r, qbin)
cuda.synchronize()

fq_grad_position2[(bpg, bpg), (tpb, tpb), stream](dq_over_r, dr)
cuda.synchronize()
fq_grad_position3[(bpg, bpg), (tpb, tpb), stream](dcos_term, dtzr)
cuda.synchronize()
fq_grad_position4[(bpg, bpg), (tpb, tpb), stream](dcos_term, dq_over_r)
cuda.synchronize()
fq_grad_position5[(bpg, bpg), (tpb, tpb), stream](dcos_term, dnorm)
cuda.synchronize()
fq_grad_position6[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
cuda.synchronize()
fq_grad_position7[(bpg, bpg), (tpb, tpb), stream](dcos_term, dfq)
cuda.synchronize()

fq_grad_position_final1[(bpg, bpg), (tpb, tpb), stream](dgrad_p, drgrad)
cuda.synchronize()
fq_grad_position_final2[(bpg, bpg), (tpb, tpb), stream](dgrad_p, dcos_term)
dgrad_p.to_host(stream)




#sum down to 1D array
grad_p=grad_p.sum(axis=(1))

dnorm.to_host(stream)

#sum reduce to 1D
na = norm_array.sum(axis=(0, 1))
na *= 1. / (scatter_array.shape[0] ** 2)
old_settings = np.seterr(all='ignore')
for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :qmin_bin] = 0.0
            grad_p[tx, tz] = np.nan_to_num(
                1 / (n * na) * grad_p[tx, tz])
np.seterr(**old_settings)
print(time()-s)
# ''', sort='cumtime')
