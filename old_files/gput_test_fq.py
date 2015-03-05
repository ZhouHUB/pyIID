__author__ = 'christopher'
# cProfile.run('''
from copy import deepcopy as dc

from pyiid.kernels.numbapro_cuda_kernels import *

# from pyiid.serial_kernel import get_normalization_array

atomsio = aseio.read(
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz')
atoms = dc(atomsio)
# for i in range(10):
# atoms += atomsio
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
# print len(scatter_array)
# get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
scatter_array = np.loadtxt(
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt')
print scatter_array.shape
scatter_array = np.asarray(scatter_array, dtype=np.float32)
atoms.set_array('scatter', scatter_array)

d = np.zeros((n, n, 3), dtype=np.float32)
r = np.zeros((n, n), dtype=np.float32)
norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
# get_normalization_array(norm_array, scatter_array)

super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)

stream = cuda.stream()
tpb = 32
bpg = int(math.ceil(float(n) / tpb))

s = time()
# push empty d, full q, and number n to GPU


norm_array2 = np.zeros((n, n, qmax_bin), dtype=np.float32)
dscat = cuda.to_device(scatter_array, stream)
dnorm = cuda.to_device(norm_array2)

get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)

dd = cuda.to_device(d, stream)
dq = cuda.to_device(q, stream)
dr = cuda.to_device(r, stream)
dfq = cuda.to_device(super_fq, stream)

get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
cuda.synchronize()

get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)

cuda.synchronize()
dr.to_host(stream)
get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
cuda.synchronize()

get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
cuda.synchronize()
dnorm.to_host()
print norm_array2
# print norm_array2

get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
cuda.synchronize()
# dfq.to_host(stream)

get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
cuda.synchronize()
# print super_fq[0, 1, 3] * norm_array[0, 1, 3]
dfq.to_host(stream)
# print super_fq[0,1,3]


#sum down to 1D array
# super_fq *= norm_array
fq = super_fq.sum(axis=(0, 1))
# print fq

# dnorm.to_host(stream)

#sum reduce to 1D
na = norm_array2.sum(axis=(0, 1))

na *= 1. / (scatter_array.shape[0] ** 2)
old_settings = np.seterr(all='ignore')
fq = np.nan_to_num(1 / (n * na) * fq)
np.seterr(**old_settings)
print(time() - s)

print fq

# print np.max(scatter_array)
# print scatter_array
# print na[:]
# print scatter_array[0, 1] * scatter_array[0, 1]
# print na.dtype
plt.plot(fq)
plt.show()
# ''', sort='tottime')