__author__ = 'christopher'
from numba import *
import math
from pyiid.kernels.flat_kernel import cuda_ij_to_k, cuda_k_to_ij

# TODO: break this up to get speed up
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


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], u4, u4])
def fast_flat_sum(new_grad, grad, k_cov, k_max):
    i, j, qx = cuda.grid(3)
    n = len(new_grad)
    if i == j or i >= n or j >= n or qx >= grad.shape[2]:
        return
    k = -1
    alpha = 0
    if j < i:
        k = cuda_ij_to_k(i, j)
        alpha = -1
    elif i < j:
        k = cuda_ij_to_k(j, i)
        alpha = 1
    for tz in range(3):
        if k_cov <= k < k_cov + k_max:
            new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
