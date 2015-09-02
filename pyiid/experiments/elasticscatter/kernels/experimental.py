__author__ = 'christopher'
import math

from numba import *

from gpu_flat import cuda_k_to_ij


@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def get_normalization_array(norm_array, scat, offset):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNx3 array
        Normalization array
    scatter_array: NxM array
        The scatter factor array
    """
    # snorm = cuda.shared.array((1, 64), f4)
    snormi = cuda.shared.array((1, 64), f4)
    snormj = cuda.shared.array((1, 64), f4)
    k, qx = cuda.grid(2)
    n = norm_array.shape[0]
    qmax_bin = norm_array.shape[1]
    if k >= n or qx >= qmax_bin:
        return
    tid = cuda.threadIdx.y
    i, j = cuda_k_to_ij(int32(k + offset))
    snormi[0, tid] = scat[i, qx]
    snormj[0, tid] = scat[j, qx]
    cuda.syncthreads()

    snormi[0, tid] *= snormj[0, tid]
    cuda.syncthreads()
    # norm_array[k, qx] = scat[i, qx] * scat[j, qx]
    norm_array[k, qx] = snormi[0, tid]


# A[k, q] = norm*Q, B[k, q] = cos(Q*r), C[k, w] = d/r/r
# D[k, q] = A*B - F(Q)
# E[k, w, q] = D * C
@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4])
def get_grad_fq_a(A, norm, qbin):
    k, qx = cuda.grid(2)
    if k >= len(A) or qx >= A.shape[1]:
        return
    A[k, qx] = norm[k, qx] * float32(qx * qbin)


@cuda.jit(argtypes=[f4[:, :], f4[:], f4])
def get_grad_fq_b(B, r, qbin):
    k, qx = cuda.grid(2)
    if k >= len(B) or qx >= B.shape[1]:
        return
    B[k, qx] = math.cos(float32(qx * qbin) * r[k])


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def get_grad_fq_c(r, d):
    k = cuda.grid(1)
    if k >= len(r):
        return
    for w in range(3):
        d[k, w] /= r[k] ** 2


# @cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :], f4[:, :]])
@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_grad_fq_d(A, B, fq):
    k, qx = cuda.grid(2)
    if k >= len(A) or qx >= A.shape[1]:
        return
    # D[k, qx] = A[k, qx] * B[k, qx] - fq[k, qx]
    A[k, qx] *= B[k, qx]
    A[k, qx] -= fq[k, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:, :]])
def get_grad_fq_e(E, D, C):
    k, qx = cuda.grid(2)
    if k >= len(E) or qx >= E.shape[2]:
        return
    for w in range(3):
        E[k, w, qx] = D[k, qx] * C[k, w]

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def experimental_sum_grad_fq1(new_grad, grad, k_cov):
    k, qx = cuda.grid(2)
    if k >= len(grad) or qx >= grad.shape[2]:
        return
    i, j = cuda_k_to_ij(i4(k + k_cov))
    for tz in range(3):
        new_grad[i, tz, qx] -= grad[k, tz, qx]
        new_grad[j, tz, qx] += grad[k, tz, qx]
    # cuda.atomic.add(new_grad, (i, 0, qx), grad[k, 0, qx] * -1)
    # cuda.atomic.add(new_grad, (j, 0, qx), grad[k, 0, qx] * 1)
    #
    # cuda.atomic.add(new_grad, (i, 1, qx), grad[k, 1, qx] * -1)
    # cuda.atomic.add(new_grad, (j, 1, qx), grad[k, 1, qx] * 1)
    #
    # cuda.atomic.add(new_grad, (i, 2, qx), grad[k, 2, qx] * -1)
    # cuda.atomic.add(new_grad, (j, 2, qx), grad[k, 2, qx] * 1)

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def experimental_sum_grad_fq2(new_grad, grad, k_cov):
    k, qx = cuda.grid(2)
    if k >= len(grad) or qx >= grad.shape[2]:
        return
    i, j = cuda_k_to_ij(i4(k + k_cov))
    for tz in range(3):
        cuda.atomic.add(new_grad, (j, tz, qx), 1)
        # new_grad[i, tz, qx] = j
        # cuda.atomic.add(new_grad, (i, tz, qx), j)

