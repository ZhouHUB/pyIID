__author__ = 'christopher'
from numba import *
from numbapro import cuda
import mkl
import math

# TODO: need more clever way to deal with this, including multiple GPUs
cuda.select_device(1)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_d_array(d, q):
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
    n = len(d)
    if tx >= n or ty >= n:
        return
    for tz in range(3):
        d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


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
    n = len(d)
    if tx >= n or ty >= n:
        return
    r[tx, ty] = math.sqrt(
        d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def get_fq_p0(fq, r, qbin):
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

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    # r is zero for tx = ty, thus we don't calculate for it
    if tx == ty:
        return

    for kq in range(0, qmax_bin):
        fq[tx, ty, kq] = kq * qbin * r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :]])
def get_fq_p1(fq):
    """
    Generate F(Q), not normalized, via the Debye sum, part 1

    Parameters:
    ---------
    fq: NxNx3 array
        The reduced scatter pattern
    """

    tx, ty = cuda.grid(2)
    n = len(fq)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(qmax_bin):
        fq[tx, ty, kq] = math.sin(fq[tx, ty, kq])


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_fq_p2(fq, r):
    """
    Generate F(Q), not normalized, via the Debye sum, part 2

    Parameters:
    ---------
    fq: NxNx3 array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(0, qmax_bin):
        fq[tx, ty, kq] /= r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def get_fq_p3(fq, norm_array):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: NxNx3 array
        The reduced scatter pattern
    norm: NxM array
        The scatter factor normalization array
    """

    tx, ty = cuda.grid(2)

    n = len(norm_array)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(qmax_bin):
        fq[tx, ty, kq] *= norm_array[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_normalization_array(norm_array, scat):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNx3 array
        Normalization array
    scatter_array: NxM array
        The scatter factor array
    """

    tx, ty = cuda.grid(2)
    n = len(scat)

    if tx >= n or ty >= n:
        return
    qmax_bin = scat.shape[1]
    for kq in range(qmax_bin):
        norm_array[tx, ty, kq] = scat[tx, kq] * scat[ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:, :]])
def fq_grad_position0(rgrad, d, r):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    rgrad: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        rgrad[tx, ty, tz] = d[tx, ty, tz] / r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def fq_grad_position1(q_over_r, r, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ----------
    q_over_r: Nx3xQ numpy array
        The array which will store the FQ gradient
    r: NxN array
        The inter-atomic distances
    qbin: float
        Scatter vector bin size
    """

    tx, ty = cuda.grid(2)
    n = len(q_over_r)
    qmax_bin = q_over_r.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        q_over_r[tx, ty, kq] = kq / r[tx, ty]
        q_over_r[tx, ty, kq] *= qbin


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def fq_grad_position3(cos_term, kqr):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    kqr: Nx3xQ array
        The array to have the cosine taken of
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] = math.cos(kqr[tx, ty, kq])


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def fq_grad_position4(cos_term, q_over_r):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    q_over_r: Nx3xQ
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] *= q_over_r[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def fq_grad_position5(cos_term, norm):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    norm: NxNxQ
        The normalization array for the configuration
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] *= norm[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def fq_grad_position6(fq, r):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ----------
    fq: Nx3xQ numpy array
        The array which will store the FQ gradient
    r: NxN
        The inter-atomic distances
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        fq[tx, ty, kq] = fq[tx, ty, kq] / r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def fq_grad_position7(cos_term, fq):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    cos_term: Nx3xQ numpy array
        The array which will store the FQ gradient
    fq: Nx3xQ array
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] -= fq[tx, ty, kq]


@cuda.jit(argtypes=[f4[:, :, :, :], f4[:, :, :]])
def fq_grad_position_final1(grad_p, rgrad):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    grad_p: NxNx3xQ numpy array
        The array which will store the FQ gradient
    rgrad: Nx3xQ array
    """

    tx, ty = cuda.grid(2)
    n = len(grad_p)
    qmax_bin = grad_p.shape[3]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        for kq in range(qmax_bin):
            grad_p[tx, ty, tz, kq] = rgrad[tx, ty, tz]


@cuda.jit(argtypes=[f4[:, :, :, :], f4[:, :, :]])
def fq_grad_position_final2(grad_p, cos_term):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    -----------
    grad_p: NxNx3xQ numpy array
        The array which will store the FQ gradient
    cos_term: Nx3xQ array
    """

    tx, ty = cuda.grid(2)
    n = len(grad_p)
    qmax_bin = grad_p.shape[3]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        for kq in range(qmax_bin):
            grad_p[tx, ty, tz, kq] *= cos_term[tx, ty, kq]