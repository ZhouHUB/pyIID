__author__ = 'christopher'
import math

import numpy as np
from numba import cuda
from numba import autojit


@autojit
def atoms_per_gpu_grad_fq(n, qmax_bin, mem):
    return int(math.floor(float(-4 * n * qmax_bin - 12 * n + .7 * mem) / (
        4 * (6 * qmax_bin * n + 3 * qmax_bin + 4 * n))))


@autojit
def atoms_per_gpu_fq(n, qmax_bin, mem):
    return int(math.floor(
        float(-4 * n * qmax_bin - 12 * n - 4 * qmax_bin + .8 * mem) / (
            8 * n * (qmax_bin + 2))))


def atomic_fq(q, scatter_array, qbin, m, n_cov):
    """
    This function wraps the GPU $F(Q)$ kernels, after the GPU has been selected

    Parameters
    ----------
    q: Nx3 array
        The atomic positions in Angstroms
    scatter_array: NxQ array
        The atomic scattering factors associated with each atom as a function
        of scattering vector
    qbin: float
        The size of each scatter vector bin
    m: int
        The number of atoms to be computed on this GPU
    n_cov: int
        Number of atoms previously covered

    Returns
    -------
    ndarray:
        The reduced scatter function F(Q)
    """
    # Create variables
    n = len(q)
    qmax_bin = scatter_array.shape[1]

    # Import and compile the GPU kernels
    from pyiid.kernels.multi_cuda import get_d_array, \
        get_normalization_array, get_r_array, \
        get_fq_step_0, get_fq_step_1, gpu_reduce_3D_to_2D, \
        gpu_reduce_2D_to_1D, zero_3D

    stream = cuda.stream()
    stream2 = cuda.stream()

    # four kinds of test_kernels; Q, NxQ, NxN, or NxNxQ
    # Q
    elements_per_dim_1 = [qmax_bin]
    tpb_l_1 = [32]
    bpg_l_1 = []
    for e_dim, tpb in zip(elements_per_dim_1, tpb_l_1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_1.append(bpg)

    # NxQ
    elements_per_dim_2_q = [n, qmax_bin]
    tpb_l_2_q = [32, 32]
    bpg_l_2_q = []
    for e_dim, tpb in zip(elements_per_dim_2_q, tpb_l_2_q):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2_q.append(bpg)

    # NXN
    elements_per_dim_2 = [m, n]
    tpb_l_2 = [32, 32]
    # tpb_l_2 = [8, 4]
    bpg_l_2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2.append(bpg)

    # NxNxQ
    elements_per_dim_3 = [m, n, qmax_bin]
    tpb_l_3 = [16, 16, 4]
    # tpb_l_3 = [4, 4, 2]
    bpg_l_3 = []
    for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_3.append(bpg)

    # start calculations
    dscat = cuda.to_device(scatter_array, stream2)
    dnorm = cuda.device_array((m, n, qmax_bin), dtype=np.float32,
                              stream=stream2)
    get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)

    dd = cuda.device_array((m, n, 3), dtype=np.float32, stream=stream)
    dr = cuda.device_array((m, n), dtype=np.float32, stream=stream)

    # Note that while direct allocation might be faster current kernels
    # depend on having zero values thus needing the zeroing of arrays

    dfq = cuda.device_array((m, n, qmax_bin), dtype=np.float32, stream=stream)
    zero_3D[bpg_l_3, tpb_l_3, stream](dfq)
    dq = cuda.to_device(q, stream)

    get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
    get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final, stream2)

    final2d = np.zeros((n, qmax_bin), dtype=np.float32)
    dfinal2d = cuda.to_device(final2d, stream2)

    get_fq_step_0[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
    get_fq_step_1[bpg_l_3, tpb_l_3, stream](dfq, dnorm)

    gpu_reduce_3D_to_2D[bpg_l_2_q, tpb_l_2_q, stream](dfinal2d, dfq)
    gpu_reduce_2D_to_1D[bpg_l_1, tpb_l_1, stream](dfinal, dfinal2d)

    dfinal.to_host(stream)
    # clear the GPU memory after transfer back to host
    del dscat, dnorm, dd, dq, dr, dfq, dfinal
    return final



def atomic_grad_fq(q, scatter_array, qbin, m, n_cov):
    """
    This function wraps the GPU $\vec{\nabla}F(Q)$ kernels,
    after the GPU has been selected

    Parameters
    ----------
    q: Nx3 array
        The atomic positions in Angstroms
    scatter_array: NxQ array
        The atomic scattering factors associated with each atom as a function
        of scattering vector
    qbin: float
        The size of each scatter vector bin
    m: int
        The number of atoms to be computed on this GPU
    n_cov: int
        Number of atoms previously covered

    Returns
    -------
    ndarray:
        The gradient of the reduced scatter function F(Q)
    """
    n = len(q)
    qmax_bin = scatter_array.shape[1]

    # F(Q) kernels
    from pyiid.kernels.multi_cuda import get_d_array, \
        get_normalization_array, get_r_array, \
        get_fq_step_0, get_fq_step_1, zero_3D
    # Grad kernels
    from pyiid.kernels.multi_cuda import fq_grad_step_0, \
        fq_grad_step_1, fq_grad_step_2, fq_grad_step_3, \
        fq_grad_step_4, gpu_reduce_4D_to_3D, zero_4D

    # cuda stream init
    stream = cuda.stream()
    stream2 = cuda.stream()
    stream3 = cuda.stream()

    # four kinds of test_kernels; Q, NxQ, NxN, or NxNxQ
    # Q
    elements_per_dim_1 = [qmax_bin]
    tpb_l_1 = [32]
    bpg_l_1 = []
    for e_dim, tpb in zip(elements_per_dim_1, tpb_l_1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_1.append(bpg)

    # NxQ
    elements_per_dim_2_q = [n, qmax_bin]
    tpb_l_2_q = [32, 32]
    bpg_l_2_q = []
    for e_dim, tpb in zip(elements_per_dim_2_q, tpb_l_2_q):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2_q.append(bpg)

    # NXN
    elements_per_dim_2 = [m, n]
    tpb_l_2 = [32, 32]
    # tpb_l_2 = [8, 4]
    bpg_l_2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2.append(bpg)

    # NxNxQ
    elements_per_dim_3 = [m, n, qmax_bin]
    tpb_l_3 = [16, 16, 4]
    # tpb_l_3 = [4, 4, 2]
    bpg_l_3 = []
    for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_3.append(bpg)

    # start F(Q) calculations
    dscat = cuda.to_device(scatter_array, stream2)
    dnorm = cuda.device_array((m, n, qmax_bin), dtype=np.float32, stream=stream2)
    get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)

    dd = cuda.device_array((m, n, 3), dtype=np.float32, stream=stream)
    dr = cuda.device_array((m, n), dtype=np.float32, stream=stream)

    # Note that while direct allocation might be faster current kernels
    # depend on having zero values thus needing the zeroing of arrays

    dfq = cuda.device_array((m, n, qmax_bin), dtype=np.float32, stream=stream)
    zero_3D[bpg_l_3, tpb_l_3, stream](dfq)
    dq = cuda.to_device(q, stream)

    get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
    get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

    get_fq_step_0[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
    get_fq_step_1[bpg_l_3, tpb_l_3, stream](dfq, dnorm)

    dcos_term = cuda.device_array((m, n, qmax_bin), dtype=np.float32,
                                  stream=stream2)

    fq_grad_step_0[bpg_l_3, tpb_l_3, stream3](dcos_term, dr, qbin)
    dgrad_p = cuda.device_array((m, n, 3, qmax_bin), dtype=np.float32,
                                stream=stream2)
    zero_4D[bpg_l_3, tpb_l_3, stream](dgrad_p)
    final = np.zeros((m, 3, qmax_bin), dtype=np.float32)

    dfinal = cuda.to_device(final, stream=stream2)

    fq_grad_step_3[bpg_l_3, tpb_l_3, stream](dgrad_p, dd, dr)
    fq_grad_step_1[bpg_l_3, tpb_l_3, stream2](dcos_term, dnorm)

    fq_grad_step_2[bpg_l_3, tpb_l_3, stream](dcos_term, dfq, dr)

    fq_grad_step_4[bpg_l_3, tpb_l_3, stream](dgrad_p, dcos_term)

    gpu_reduce_4D_to_3D[bpg_l_3, tpb_l_3, stream](dfinal, dgrad_p)

    dfinal.to_host()
    del dscat, dnorm, dd, dr, dfq, dcos_term, dgrad_p
    return final, n_cov