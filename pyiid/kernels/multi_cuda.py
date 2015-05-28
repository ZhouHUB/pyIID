__author__ = 'christopher'
from numba import *
from numbapro import cuda
import mkl
import math
import numpy as np

# F(Q) kernels ---------------------------------------------------------------

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], i4])
def get_d_array(d, q, offset):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
        The inter-atomic x, y, z distances
    q: Nx3 array
        The atomic positions
    """
    tx, ty = cuda.grid(2)

    m = d.shape[0]
    n = d.shape[1]
    if tx >= m or ty >= n:
        return
    for tz in range(3):
        d[tx, ty, tz] = q[ty, tz] - q[tx + offset, tz]


@cuda.jit(argtypes=[f4[:, :], f4[:, :, :]])
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
        Pair distances
    d: NxNx3 array
        The coordinate pair distances
    n: 1d array
        Range of atomic numbers
    """
    tx, ty = cuda.grid(2)

    m = r.shape[0]
    n = r.shape[1]

    if tx >= m or ty >= n:
        return
    r[tx, ty] = math.sqrt(
        d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def get_fq_step_0(fq, r, qbin):
    """
    Get part of the reduced structure factor.
    The FQ calculation is broken up because of GPU register issues

    Parameters
    -----------
    fq: NxNx3 zero array
    r: NxN array
        Holds the pair distances
    qbin: float
        The size of the scatter vector bins
    """

    tx, ty, kq = cuda.grid(3)

    m = fq.shape[0]
    n = fq.shape[1]
    qmax_bin = fq.shape[2]
    # r is zero for tx = ty, thus we don't calculate for it
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    fq[tx, ty, kq] = math.sin(kq * qbin * r[tx, ty]) / r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def get_fq_step_1(fq, norm_array):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: NxNx3 array
        The reduced scatter pattern
    norm: NxM array
        The scatter factor normalization array
    """

    tx, ty, kq = cuda.grid(3)

    m = fq.shape[0]
    n = fq.shape[1]
    qmax_bin = fq.shape[2]
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    fq[tx, ty, kq] *= norm_array[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], i4])
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

    tx, ty, kq = cuda.grid(3)

    m = norm_array.shape[0]
    n = norm_array.shape[1]
    qmax_bin = norm_array.shape[2]

    if tx >= m or ty >= n or kq >= qmax_bin:
        return
    norm_array[tx, ty, kq] = scat[tx + offset, kq] * scat[ty, kq]

# Gradient kernels -----------------------------------------------------------

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def fq_grad_step_0(cos_term, r, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    qbin:
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    kqr: Nx3xQ array
        The array to have the cosine taken of
    """

    tx, ty, kq = cuda.grid(3)

    m = cos_term.shape[0]
    n = cos_term.shape[1]
    qmax_bin = cos_term.shape[2]
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    cos_term[tx, ty, kq] = math.cos(kq * qbin * r[tx, ty]) * kq * qbin / r[
        tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def fq_grad_step_1(cos_term, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    norm: NxNxQ
        The normalization array for the configuration
    """

    tx, ty, kq = cuda.grid(3)

    m = cos_term.shape[0]
    n = cos_term.shape[1]
    qmax_bin = cos_term.shape[2]
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    cos_term[tx, ty, kq] *= norm[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:, :]])
def fq_grad_step_2(cos_term, fq, r):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    fq: Nx3xQ array
    """

    tx, ty, kq = cuda.grid(3)

    m = cos_term.shape[0]
    n = cos_term.shape[1]
    qmax_bin = cos_term.shape[2]
    if tx >= m or ty >= n or kq >= qmax_bin:
        return
    if tx == ty:
        return
    cos_term[tx, ty, kq] -= fq[tx, ty, kq] / r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :, :], f4[:, :, :], f4[:, :]])
def fq_grad_step_3(grad_p, d, r):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    grad_p: NxNx3xQ numpy array
        The array which will store the FQ gradient
    rgrad: Nx3xQ array
    """

    tx, ty, kq = cuda.grid(3)

    m = grad_p.shape[0]
    n = grad_p.shape[1]
    qmax_bin = grad_p.shape[3]
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    for tz in range(3):
        grad_p[tx, ty, tz, kq] = d[tx, ty, tz] / r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :, :], f4[:, :, :]])
def fq_grad_step_4(grad_p, cos_term):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    grad_p: NxNx3xQ numpy array
        The array which will store the FQ gradient
    cos_term: Nx3xQ array
    """

    tx, ty, kq = cuda.grid(3)

    m = grad_p.shape[0]
    n = grad_p.shape[1]
    qmax_bin = grad_p.shape[3]
    if tx >= m or ty >= n or kq >= qmax_bin or tx == ty:
        return
    for tz in range(3):
        grad_p[tx, ty, tz, kq] *= cos_term[tx, ty, kq]

# General Kernels ------------------------------------------------------------
@cuda.jit(argtypes=[f4[:], f4[:, :, :]])
def gpu_reduce_3D_to_1D(reduced, A):

    threadi = cuda.grid(1)

    if threadi >= A.shape[2]:
        return
    reduced[threadi] = A[:, :, threadi].sum()

@cuda.jit(argtypes=[f4[:, :], f4[:, :, :]])
def gpu_reduce_3D_to_2D(reduced, A):

    threadi, threadj = cuda.grid(2)

    if threadi >= A.shape[0] or threadj >= A.shape[2]:
        return
    reduced[threadi, threadj] = A[threadi, :, threadj].sum()

@cuda.jit(argtypes=[f4[:], f4[:, :]])
def gpu_reduce_2D_to_1D(reduced, A):

    threadi = cuda.grid(1)

    if threadi >= A.shape[1]:
        return

    N = A.shape[0]

    tmp = 0.0
    # reduced[threadi] = 0.0
    for k in range(N):
        tmp += A[k, threadi]
        # reduced[threadi] += A[k, threadi]
    reduced[threadi] = tmp


@cuda.jit(argtypes=[f4[:,:,:], f4[:, :, :, :]])
def gpu_reduce_4D_to_3D(reduced, A):

    imax, _, jmax, kmax = A.shape
    i, j, k = cuda.grid(3)

    if i >= imax or j >= jmax or k >= kmax:
        return
    reduced[i, j, k] = A[i, :, j, k].sum()