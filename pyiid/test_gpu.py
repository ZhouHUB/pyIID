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
from copy import deepcopy as dc
from pyiid.wrappers.gpu_wrap import wrap_fq_gpu, wrap_fq_grad_gpu
from pyiid.serial_kernel import get_d_array, get_r_array
import math

atomsio = aseio.read(
    '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz')
scatter_array = np.loadtxt('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt', dtype=np.float32)
atomsio.set_array('scatter', scatter_array)
atoms = dc(atomsio)
# for i in range(10):
#     atoms += atomsio
print scatter_array.shape

wrap_fq_gpu(atoms)

grad = wrap_fq_grad_gpu(atoms)
print grad
tx = 1
# ty = 2
tz = 2
kq = 5
qbin = .1


q= atoms.get_positions()
symbols = atoms.get_chemical_symbols()

# define scatter_q information and initialize constants
n = len(q)

# Get pair coordinate distance array
d = np.zeros((n, n, 3))
get_d_array(d, q, n)

# Get pair distance array
r = np.zeros((n, n))
get_r_array(r, d, n)

sub_grad_p = 0
for ty in range(n):
    if tx != ty:
        sub_grad_p += scatter_array[tx, kq] * scatter_array[ty, kq] * d[tx, ty, tz] * \
                            ((kq * qbin) * r[tx, ty] *
                                math.cos(kq * qbin * r[tx, ty]) -
                                math.sin(kq * qbin * r[tx, ty])
                            ) / (r[tx, ty] ** 3)

# print sub_grad_p


tx = 1
ty = 2
tz = 2
kq = 5
qbin = .1
p0 = d[tx, ty, tz]/r[tx, ty]
# print 'p0, cpu', p0
p1 = kq/r[tx, ty]*qbin
# print 'p1, cpu', p1
p3 = math.cos(kq*qbin*r[tx,ty])
print 'cpu kqr', kq*qbin*r[tx,ty]
print 'p3, cpu', p3
p4 = p3 * p1
# print sub_grad_p