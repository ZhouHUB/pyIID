__author__ = 'christopher'
from numba import *
import math
import numpy as np


def ij_to_k(i, j):
    return int(j + i * (i - 1) / 2)


def k_to_ij(k):
    i = math.floor(float((1 + math.sqrt(1 + 8. * k))) / 2.)
    j = k - i * (i - 1) / 2.
    return int(i), int(j)


@cuda.jit(device=True)
def cuda_ij_to_k(i, j):
    return int32(j + i * (i - 1) / 2)


@cuda.jit(device=True)
def cuda_k_to_ij(k):
    i = float32(math.floor((1 + math.sqrt(float32(1 + 8. * k))) / 2.))
    j = k - i * (i - 1) / 2.
    return int32(i), int32(j)


def symmetric_reshape(out_data, in_data):
    for i in range(len(out_data)):
        for j in range(i):
            out_data[i, j] = in_data[j + i * (i - 1) / 2]
            out_data[j, i] = in_data[j + i * (i - 1) / 2]


def antisymmetric_reshape(out_data, in_data):
    for i in range(len(out_data)):
        for j in range(len(out_data)):
            if j < i:
                out_data[i, j] = -1 * in_data[ij_to_k(i, j)]
            elif i < j:
                out_data[i, j] = in_data[ij_to_k(j, i)]


# TODO: test this with grid(2)
@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def get_d_array(d, q, offset):
    k = cuda.grid(1)
    if k >= len(d):
        return
    i, j = cuda_k_to_ij(k + offset)
    for tz in range(3):
        d[k, tz] = q[i, tz] - q[j, tz]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def get_r_array(r, d):
    k = cuda.grid(1)

    if k >= len(r):
        return
    r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)


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

    k, qx = cuda.grid(2)

    n = norm_array.shape[0]
    qmax_bin = norm_array.shape[1]
    if k >= n or qx >= qmax_bin:
        return
    i, j = cuda_k_to_ij(int32(k + offset))
    norm_array[k, qx] = scat[i, qx] * scat[j, qx]


@cuda.jit(argtypes=[f4[:, :], f4[:], f4[:, :], f4])
def get_fq(fq, r, norm, qbin):
    k, qx = cuda.grid(2)
    if k >= fq.shape[0] or qx >= fq.shape[1]:
        return
    Q = float32(qbin * qx)
    rk = r[k]
    fq[k, qx] = norm[k, qx] * math.sin(Q * rk) / rk


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:], f4[:, :], f4[:, :], f4])
def get_grad_fq(grad, fq, r, d, norm, qbin):
    k, qx = cuda.grid(2)
    if k >= grad.shape[0] or qx >= grad.shape[2]:
        return
    Q = float32(qbin * qx)
    A = (norm[k, qx] * Q * math.cos(Q * r[k]) - fq[k, qx]) / float32(r[k] ** 2)
    for w in range(3):
        grad[k, w, qx] = A * d[k, w]


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def d2_to_d1_sum(d1, d2):
    qx = cuda.grid(1)

    if qx >= len(d1):
        return
    tmp = d2[:, qx].sum()
    d1[qx] = tmp


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4, i4, i4])
def fast_flat_sum(new_grad, grad, k_cov, k_max, i_min):
    i, qx = cuda.grid(2)
    i += i_min
    i = int32(i)
    n = int32(len(new_grad))
    if i >= n or qx >= grad.shape[2]:
        return
    for tz in range(3):
        tmp = float32(0.)
        for j in range(n):
            k = int32(-1)
            if j < i:
                k = cuda_ij_to_k(i, j)
                alpha = float32(-1.)
            elif j > i:
                k = cuda_ij_to_k(j, i)
                alpha = float32(1.)
            k -= k_cov
            if 0 <= k < k_max:
                tmp += float32(grad[k, tz, qx] * alpha)
        new_grad[i, tz, qx] = tmp
