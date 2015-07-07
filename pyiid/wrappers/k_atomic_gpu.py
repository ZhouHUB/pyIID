from pyiid.wrappers import generate_grid

__author__ = 'christopher'
import math
from numba import cuda
import numpy as np


def atoms_per_gpu_grad_fq(n, qmax_bin, mem):
    return int(math.floor(
        float(mem - 4 * qmax_bin * n - 12 * n) / (
            4 * (7 * qmax_bin + 12))))


def atoms_pdf_gpu_fq(n, qmax_bin, mem):
    return int(
        math.floor(float(mem - 4 * qmax_bin - 4 * qmax_bin * n - 12 * n) / (
            16 * (qmax_bin + 3))))


def atomic_fq(q, scatter_array, qbin, k_max, k_cov):
    qmax_bin = scatter_array.shape[1]

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
    tpb2 = [16, 4]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q)

    # calculate kernels

    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    get_d_array[bpg1, tpb1, stream](dd, dq, k_cov)
    del dq

    get_r_array[bpg1, tpb1, stream](dr, dd)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)

    dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)

    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscat, k_cov)
    del dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                            stream=stream2)

    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)
    del dr, dnorm
    d2_to_d1_sum[bpgq, tpbq, stream2](dfinal, dfq)
    del dfq

    dfinal.to_host(stream2)
    del dfinal
    return final


def atomic_grad_fq(q, scatter_array, qbin, k_cov, k_max):
    qmax_bin = scatter_array.shape[1]

    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, get_grad_fq, \
        zero_pseudo_3D, fast_flat_sum

    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [64]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_q = [qmax_bin]
    tpbq = [4]
    bpgq = generate_grid(elements_per_dim_q, tpbq)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [16, 4]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    elements_per_dim_nq = [len(q), qmax_bin]
    tpbnq = [16, 4]
    bpgnq = generate_grid(elements_per_dim_nq, tpbnq)

    # gnerate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q, stream=stream)

    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)

    # calculate kernels
    get_d_array[bpg1, tpb1, stream](dd, dq, k_cov)

    get_r_array[bpg1, tpb1, stream](dr, dd)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                              stream=stream2)
    dscat = cuda.to_device(scatter_array, stream=stream2)

    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscat, k_cov)
    del dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                            stream=stream2)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    grad = np.zeros((k_max, 3, qmax_bin), dtype=np.float32)
    dgrad = cuda.device_array(grad.shape, dtype=np.float32, stream=stream2)

    get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)

    new_grad2 = np.zeros((len(q), 3, qmax_bin), dtype=np.float32)

    dnew_grad = cuda.device_array(new_grad2.shape, dtype=np.float32,
                                  stream=stream2)

    fast_flat_sum[bpgnq, tpbnq](dnew_grad, dgrad, k_cov, k_max)
    dnew_grad.copy_to_host(new_grad2)
    del dd, dr, dnorm, dfq, dgrad
    del grad
    return new_grad2