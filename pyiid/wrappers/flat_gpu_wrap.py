__author__ = 'christopher'
import numpy as np
from numba import cuda
import math
from threading import Thread
from numpy.testing import assert_allclose
from pyiid.kernels.cpu_kernel import get_d_array as cpu

from pyiid.kernels.flat_kernel import antisymmetric_reshape, symmetric_reshape, \
    get_ij_lists

def subs_fq(gpu, qi, qj, scati, scatj, fq_q, qmax_bin, qbin, k_start, k_stop):
    pass


def wrap_fq(atoms, qmax=25., qbin=.1):
    # set up atoms
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(math.ceil(qmax / qbin))
    scatter_array = atoms.get_array('scatter')

    # setup flat map
    il, jl = get_ij_lists(n)
    qi = q[il, :]
    qj = q[jl]

    scati = scatter_array[il]
    scatj = scatter_array[jl]

    # set up GPU
    cuda.select_device(1)
    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, d2_to_d1_sum

    elements_per_dim_1 = [len(il)]
    tpb1 = [32]
    bpg1 = []
    for e_dim, tpb in zip(elements_per_dim_1, tpb1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg1.append(bpg)

    elements_per_dim_q = [qmax_bin]
    tpbq = [32]
    bpgq = []
    for e_dim, tpb in zip(elements_per_dim_q, tpb1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpgq.append(bpg)

    elements_per_dim_2 = [len(il), qmax_bin]
    tpb2 = [16, 2]
    bpg2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg2.append(bpg)


    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    d = np.zeros((len(il), 3), dtype=np.float32)
    dd = cuda.to_device(d, stream)
    dqi = cuda.to_device(qi, stream)
    dqj = cuda.to_device(qj, stream)
    r = np.zeros(len(il), dtype=np.float32)
    # dr = cuda.device_array(len(il), dtype=np.float32, stream)
    dr = cuda.to_device(r)

    # calculate kernels
    get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
    del dqi, dqj, qi, qj
    get_r_array[bpg1, tpb1, stream](dr, dd)

    norm = np.zeros((len(il), qmax_bin), dtype=np.float32)

    dnorm = cuda.to_device(norm)
    dscati = cuda.to_device(scati)
    dscatj = cuda.to_device(scatj)

    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)
    del dscati, dscatj, scati, scatj
    fq = np.zeros((len(il), qmax_bin), dtype=np.float32)
    dfq = cuda.to_device(fq)

    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)
    del dr, dnorm, r, norm
    d2_to_d1_sum[bpgq, tpbq, stream2](dfinal, dfq)
    del dfq, fq

    dfinal.to_host(stream2)
    del dfinal

    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    return 2 * final



def wrap_fq_grad(atoms, qmax=25, qbin=.1):
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(math.ceil(qmax / qbin))
    scatter_array = atoms.get_array('scatter')

    # setup flat map
    il, jl = get_ij_lists(n)
    qi = q[il, :]
    qj = q[jl]

    scati = scatter_array[il]
    scatj = scatter_array[jl]

    # set up GPU
    cuda.select_device(1)
    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, d2_to_d1_sum, get_grad_fq

    elements_per_dim_1 = [len(il)]
    tpb1 = [32]
    bpg1 = []
    for e_dim, tpb in zip(elements_per_dim_1, tpb1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg1.append(bpg)

    elements_per_dim_q = [qmax_bin]
    tpbq = [32]
    bpgq = []
    for e_dim, tpb in zip(elements_per_dim_q, tpb1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpgq.append(bpg)

    elements_per_dim_2 = [len(il), qmax_bin]
    tpb2 = [16, 2]
    bpg2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg2.append(bpg)


    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    d = np.zeros((len(il), 3), dtype=np.float32)
    dd = cuda.to_device(d, stream)
    dqi = cuda.to_device(qi, stream)
    dqj = cuda.to_device(qj, stream)
    r = np.zeros(len(il), dtype=np.float32)
    # dr = cuda.device_array(len(il), dtype=np.float32, stream)
    dr = cuda.to_device(r)

    # calculate kernels
    get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
    # dd.to_host(stream)

    get_r_array[bpg1, tpb1, stream](dr, dd)
    # dr.to_host(stream)

    norm = np.zeros((len(il), qmax_bin), dtype=np.float32)
    dnorm = cuda.to_device(norm)
    dscati = cuda.to_device(scati)
    dscatj = cuda.to_device(scatj)

    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)

    fq = np.zeros((len(il), qmax_bin), dtype=np.float32)
    dfq = cuda.to_device(fq)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    grad = np.zeros((len(il), 3, qmax_bin), dtype=np.float32)
    dgrad = cuda.to_device(grad, stream2)

    get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)
    dgrad.to_host(stream2)

    newgrad = np.zeros((n, n, 3, qmax_bin))
    antisymmetric_reshape(newgrad, grad, il, jl)

    newgrad = newgrad.sum(axis=1)
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            newgrad[tx, tz] = np.nan_to_num(1 / na * newgrad[tx, tz])
    np.seterr(**old_settings)

    del dd, dqi, dqj, dr, dnorm, dscati, dscatj, dfq, dgrad
    del d, qi, qj, r, norm, scati, scatj, grad, il, jl

    return newgrad


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.master_wrap import wrap_atoms
    import matplotlib.pyplot as plt

    # n = 300
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms, None)

    fq = wrap_fq(atoms)
    grad_fq = wrap_fq_grad(atoms)
    # print grad_fq