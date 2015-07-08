__author__ = 'christopher'
import math

from numba import *
import numpy as np
from ase.atoms import Atoms
from numpy.testing import assert_allclose

from pyiid.wrappers import generate_grid
from pyiid.wrappers.elasticscatter import wrap_atoms
from pyiid.kernels import ij_to_k, k_to_ij


def generate_d(d, q, offset):
    for k in xrange(len(d)):
        i, j = k_to_ij(k + offset)
        for tz in range(3):
            d[k, tz] = q[i, tz] - q[j, tz]


def generate_r(r, d):
    for k in xrange(len(r)):
        r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)


def generate_norm(norm, scat, offset):
    for k in xrange(norm.shape[0]):
        i, j = k_to_ij(k + offset)
        for qx in xrange(norm.shape[1]):
            i, j = k_to_ij(k + offset)
            norm[k, qx] = scat[i, qx] * scat[j, qx]


def generate_fq(fq, r, norm, qbin):
    for k in xrange(fq.shape[0]):
        for qx in xrange(fq.shape[1]):
            fq[k, qx] = norm[k, qx] * math.sin(float32(qbin * qx) * r[k]) / r[k]

def generate_grad_fq(grad, fq, r, d, norm, qbin):
    for k in xrange(len(grad)):
        for qx in xrange(norm.shape[1]):
            for w in range(3):
                grad[k, w, qx] = (norm[k, qx] * float32(qx * qbin) * math.cos(
                    float32(qx * qbin) * r[k]) - fq[k, qx]) / r[k] * d[k, w] / r[k]


def generate_fast_flat_sum(new_grad, grad, k_cov, k_max):
    for i in range(len(new_grad)):
        for qx in range(grad.shape[2]):
            alpha = 0
            for tz in range(3):
                tmp = 0.
                for j in range(len(new_grad)):
                    k = -1
                    if j < i:
                        k = ij_to_k(i, j)
                        alpha = -1
                    elif j > i:
                        k = ij_to_k(j, i)
                        alpha = 1
                    if k_cov <= k < k_cov + k_max:
                        tmp += grad[k - k_cov, tz, qx] * alpha
                new_grad[i, tz, qx] = tmp

def atomic_setup():
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    atoms.center(5)
    atoms *= 4
    n = len(atoms)
    wrap_atoms(atoms)
    q = atoms.get_positions().astype(np.float32)
    scatter_array = atoms.get_array('F(Q) scatter')
    qbin = atoms.info['exp']['qbin']

    k_max = int(n * (n - 1) / 2.)
    qmax_bin = scatter_array.shape[1]
    return q, scatter_array, k_max, qmax_bin, qbin


def test_fq():
    q, scatter_array, k_max, qmax_bin, qbin = atomic_setup()

    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, d2_to_d1_sum

    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [32]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_q = [qmax_bin]
    tpbq = [4]
    bpgq = generate_grid(elements_per_dim_q, tpbq)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [1, 64]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()
    # calculate kernels

    # test dd
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q)

    get_d_array[bpg1, tpb1, stream](dd, dq, 0)

    gpu_d = np.empty(dd.shape, np.float32)
    dd.copy_to_host(gpu_d)
    d = np.empty(dd.shape)
    generate_d(d, q, 0)
    assert_allclose(gpu_d, d)

    # test dr
    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    get_r_array[bpg1, tpb1, stream](dr, dd)

    gpu_r = np.empty(dr.shape, np.float32)
    dr.copy_to_host(gpu_r)

    r = np.empty(dr.shape)
    generate_r(r, d)

    r2 = np.empty(dr.shape)
    generate_r(r2, gpu_d)

    # GPU-GPU VS CPU-CPU
    assert_allclose(gpu_r, r)
    # GPU-GPU VS GPU-CPU
    assert_allclose(gpu_r, r)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                              stream=stream2)
    dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)

    # test dnorm
    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscat, 0)

    gpu_norm = np.empty(dnorm.shape, np.float32)
    dnorm.copy_to_host(gpu_norm)
    norm = np.empty(dnorm.shape, np.float32)
    generate_norm(norm, scatter_array, 0)
    assert_allclose(gpu_norm, norm)

    del dscat
    # test F(Q)
    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                            stream=stream2)
    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    gpu_fq = np.empty(dfq.shape, np.float32)
    dfq.copy_to_host(gpu_fq)
    fq = np.empty(dfq.shape, np.float32)
    generate_fq(fq, r, norm, qbin)

    fq2 = np.empty(dfq.shape, np.float32)
    generate_fq(fq2, gpu_r, gpu_norm, qbin)


    assert_allclose(gpu_fq, fq2, rtol=1e-5, atol=5e-4)

    assert_allclose(gpu_fq, fq, atol=5e-4)



    del dr, dnorm
    d2_to_d1_sum[bpgq, tpbq, stream2](dfinal, dfq)
    del dfq

    dfinal.to_host(stream2)
    del dfinal
    assert_allclose(final, np.sum(fq, axis=0))


def test_grad_fq():
    q, scatter_array, k_max, qmax_bin, qbin = atomic_setup()

    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, get_grad_fq, \
        antisymmetric_reshape

    n = len(q)
    k_cov = 0
    i_min, _ = k_to_ij(k_cov)
    i_max, _ = k_to_ij(k_cov + k_max)
    print i_min, i_max
    print k_cov, k_max
    print ij_to_k(n - 1, n - 1)
    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [64]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [1, 64]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    elements_per_dim_nq = [i_max - i_min + 1, qmax_bin]
    # elements_per_dim_nq = [len(q), qmax_bin]
    tpbnq = [1, 64]
    bpgnq = generate_grid(elements_per_dim_nq, tpbnq)


    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # Start calculations
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q, stream=stream)

    get_d_array[bpg1, tpb1, stream](dd, dq, k_cov)
    gpu_d = np.empty(dd.shape, np.float32)
    dd.copy_to_host(gpu_d)

    d = np.empty(dd.shape)
    generate_d(d, q, 0)
    assert_allclose(gpu_d, d)

    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    get_r_array[bpg1, tpb1, stream](dr, dd)

    gpu_r = np.empty(dr.shape, np.float32)
    dr.copy_to_host(gpu_r)

    r = np.empty(dr.shape)
    generate_r(r, d)

    r2 = np.empty(dr.shape)
    generate_r(r2, gpu_d)

    # GPU-GPU VS CPU-CPU
    assert_allclose(gpu_r, r)
    # GPU-GPU VS GPU-CPU
    assert_allclose(gpu_r, r)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                              stream=stream2)
    dscat = cuda.to_device(scatter_array, stream=stream2)

    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscat, k_cov)
    gpu_norm = np.empty(dnorm.shape, np.float32)
    dnorm.copy_to_host(gpu_norm)
    norm = np.empty(dnorm.shape, np.float32)
    generate_norm(norm, scatter_array, 0)
    assert_allclose(gpu_norm, norm)
    del dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                            stream=stream2)
    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)
    fq = np.zeros(dfq.shape, dtype=np.float32)
    dfq.copy_to_host(fq)

    grad = np.zeros((k_max, 3, qmax_bin), dtype=np.float32)
    cpu_grad = grad.copy()
    dgrad = cuda.device_array(grad.shape, dtype=np.float32, stream=stream2)
    print [a * b for a, b in zip(tpb2, bpg2)]
    get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)

    del dd, dr, dnorm, dfq
    dgrad.copy_to_host(grad)


    generate_grad_fq(cpu_grad, fq, r, d, norm, qbin)
    assert_allclose(grad, cpu_grad)

    # new_grad2 = np.zeros((len(q), 3, qmax_bin), dtype=np.float32)
    new_grad2 = np.zeros((len(q), len(q), 3, qmax_bin), dtype=np.float32)

    antisymmetric_reshape(new_grad2, grad)

    # dnew_grad = cuda.device_array(new_grad2.shape, dtype=np.float32,
    #                               stream=stream2)
    # fast_flat_sum[bpgnq, tpbnq, stream2](dnew_grad, dgrad, k_cov, k_max,
    #                                      i_min - 1)
    # dnew_grad.copy_to_host(new_grad2)
    # del dgrad, dnew_grad
    # del grad
    # return new_grad2
    z = np.sum(new_grad2, axis=1)
    print z.shape


if __name__ == '__main__':
    # import nose

    # nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
    # raw_input()
    test_grad_fq()