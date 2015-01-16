__author__ = 'christopher'

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
print n
qmin_bin = 0
qmax_bin = int(qmax / qbin)

# Atoms definition, outside of calc

# scatter_array = np.zeros((n, qmax_bin))
scatter_array = np.ones((n, qmax_bin), dtype=np.float32) * 2
# get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
atoms.set_array('scatter', scatter_array)

d = np.zeros((n, n, 3), dtype=np.float32)
r = np.zeros((n, n), dtype=np.float32)
norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)

# scatter_array = np.zeros((n, qmax_bin))
scatter_array = np.ones((n, qmax_bin), dtype=np.float32) * 2
# get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
atoms.set_array('scatter', scatter_array)

super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)
# print super_fq.shape

stream = cuda.stream()
tpb = 32
bpg = int(math.ceil(float(n) / tpb))
print(bpg, tpb, bpg * tpb)

s = time()
#push empty d, full q, and number n to GPU
dd = cuda.to_device(d, stream)
dq = cuda.to_device(q, stream)
dr = cuda.to_device(r, stream)
dfq = cuda.to_device(super_fq, stream)
dscat = cuda.to_device(scatter_array, stream)
# dqbin = cuda.to_device(np.asarray(qbin, dtype=np.float32))
dnorm = cuda.to_device(norm_array)

get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
cuda.synchronize()

get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)
cuda.synchronize()

get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
cuda.synchronize()

get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)
cuda.synchronize()

get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
cuda.synchronize()

get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
cuda.synchronize()

dfq.to_host(stream)


#sum down to 1D array
fq = super_fq.sum(axis=(0, 1))
dnorm.to_host(stream)

#sum reduce to 1D
na = norm_array.sum(axis=(0, 1))

na *= 1. / (scatter_array.shape[0] ** 2)
old_settings = np.seterr(all='ignore')
fq = np.nan_to_num(1 / (n * na) * fq)
np.seterr(**old_settings)

print time() - s
# print(fq.shape)
# print fq

plt.plot(fq)
plt.show()


