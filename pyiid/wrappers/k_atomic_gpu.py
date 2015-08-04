from pyiid.kernels import k_to_ij
from pyiid.wrappers import generate_grid

__author__ = 'christopher'
import math
from numba import *
import numpy as np
from copy import deepcopy as dc


def atoms_pdf_gpu_fq(n, qmax_bin, mem):
    return int(math.floor(
        float(.8 * mem - 4 * qmax_bin - 4 * qmax_bin * n - 12 * n) / (
            8 * (qmax_bin + 2))))


def atoms_per_gpu_grad_fq(n, qmax_bin, mem):
    return int(math.floor(
        float(.8 * mem - 16 * qmax_bin * n - 12 * n) / (
            4 * (5 * qmax_bin + 4))))


def atomic_fq(q, scatter_array, qbin, k_max, k_cov):
    qmax_bin = scatter_array.shape[1]

    # load kernels
    from pyiid.kernels.flat_kernel import (get_d_array,
                                           get_r_array,
                                           get_normalization_array,
                                           get_fq,
        d2_to_d1_sum
                                           )
    from pyiid.kernels.experimental_kernels import experimental_sum_fq, \
        d2_to_d1_cleanup_kernel, d2_zero
    # generate grids
    elements_per_dim_1 = [k_max]
    tpb_k = [32]
    bpg_k = generate_grid(elements_per_dim_1, tpb_k)

    elements_per_dim_q = [qmax_bin]
    tpb_q = [4]
    bpg_q = generate_grid(elements_per_dim_q, tpb_q)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb_kq = [1, 128]
    bpg_kq = generate_grid(elements_per_dim_2, tpb_kq)

    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # calculate kernels
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q, stream=stream)
    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    get_d_array[bpg_k, tpb_k, stream](dd, dq, k_cov)
    del dq
    get_r_array[bpg_k, tpb_k, stream](dr, dd)
    del dd

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)
    get_normalization_array[bpg_kq, tpb_kq, stream2](dnorm, dscat, k_cov)
    del dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    get_fq[bpg_kq, tpb_kq, stream2](dfq, dr, dnorm, qbin)
    del dr, dnorm

    # '''
    # summation kernel
    tpb_sum = [512, 1]
    bpg_sum = generate_grid(elements_per_dim_2, tpb_sum)

    # Generate array to hold summation
    dsum = cuda.device_array((bpg_sum[0], qmax_bin), np.float32)
    # Generate grid to zero the summation array, which is smaller
    bpg_sum2 = generate_grid(bpg_sum, tpb_sum)

    # Zero the summation array
    d2_zero[bpg_sum2, tpb_sum, stream2](dsum)

    # Sum
    experimental_sum_fq[bpg_sum, tpb_sum, stream2](dsum, dfq, k_max)

    # We may need to sum more than once so sum as much as needed
    while bpg_sum[0] > 1:
        bpg_sum = generate_grid(bpg_sum, tpb_sum)
        dsum2 = cuda.device_array((bpg_sum[0], qmax_bin), np.float32, stream=stream2)
        d2_zero[bpg_sum, tpb_sum, stream2](dsum2)
        experimental_sum_fq[bpg_sum, tpb_sum, stream2](dsum2, dsum, dsum.shape[0])
        dsum = dsum2
        # del dsum2

    dfinal = cuda.device_array(qmax_bin, np.float32, stream=stream2)
    d2_to_d1_cleanup_kernel[bpg_q, tpb_q, stream2](dfinal, dsum)
    final = dfinal.copy_to_host(stream=stream2)

    tst = False
    # tst = True
    if tst is True:
        # test_sum = dsum2.copy_to_host()
        # print test_sum[1, :]

        test_fq = dfq.copy_to_host()
        np.testing.assert_allclose(final, test_fq.sum(0), rtol=1e-4, atol=1e-5)
    del dsum, dfinal
    return final
    '''
    dfinal = cuda.device_array(qmax_bin, np.float32, stream=stream2)
    d2_to_d1_sum[bpg_q, tpb_q, stream2](dfinal, dfq)
    final = dfinal.copy_to_host(stream=stream2)
    del dq, dd, dscat, dr, dnorm, dfq, dfinal
    return final
    # '''



def atomic_grad_fq(q, scatter_array, qbin, k_cov, k_max):
    qmax_bin = scatter_array.shape[1]

    # load kernels
    from pyiid.kernels.flat_kernel import (get_d_array, get_r_array,
                                           get_normalization_array,
                                           get_fq,
                                           get_grad_fq, zero3d,
                                           fast_fast_flat_sum
                                           )
    # from pyiid.kernels.experimental_kernels import get_normalization_array
    n = len(q)
    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [64]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [1, 64]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    elements_per_dim_nq = [n, qmax_bin]
    tpbnq = [1, 64]
    bpgnq = generate_grid(elements_per_dim_nq, tpbnq)

    elements_per_dim_nnq = [n, n, qmax_bin]
    tpbnnq = [1, 1, 64]
    bpgnnq = generate_grid(elements_per_dim_nnq, tpbnnq)

    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # Start calculations
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    # dd = cuda.to_device(np.zeros((k_max, 3), dtype=np.float32), stream=stream)
    dq = cuda.to_device(q, stream=stream)
    get_d_array[bpg1, tpb1, stream](dd, dq, k_cov)
    # del dq

    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    # dr = cuda.to_device(np.zeros(k_max, dtype=np.float32), stream=stream)
    get_r_array[bpg1, tpb1, stream](dr, dd)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                              stream=stream2)
    # dnorm = cuda.to_device(np.zeros((k_max, qmax_bin), dtype=np.float32), stream=stream2)
    dscat = cuda.to_device(scatter_array, stream=stream2)
    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscat, k_cov)
    # del dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32,
                            stream=stream2)
    # dfq = cuda.to_device(np.zeros((k_max, qmax_bin), dtype=np.float32), stream=stream2)
    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    dgrad = cuda.device_array((k_max, 3, qmax_bin), dtype=np.float32,
                              stream=stream2)
    # dgrad = cuda.to_device(np.zeros((k_max, 3, qmax_bin), dtype=np.float32), stream=stream2)
    get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)
    # del dd, dr, dnorm, dfq

    new_grad2 = np.zeros((len(q), 3, qmax_bin), dtype=np.float32)
    dnew_grad = cuda.device_array(new_grad2.shape, dtype=np.float32,
                                  stream=stream2)
    # dnew_grad = cuda.to_device(new_grad2, stream=stream2)

    zero3d[bpgnnq, tpbnnq, stream2](dnew_grad)
    fast_fast_flat_sum[bpgnnq, tpbnnq, stream2](dnew_grad, dgrad, k_cov)

    # fast_fast_flat_sum[bpgnq, tpbnq, stream2](dnew_grad, dgrad, k_cov)
    dnew_grad.copy_to_host(new_grad2, stream=stream2)
    del dgrad, dnew_grad
    del dq, dscat, dd, dr, dnorm, dfq
    return new_grad2
