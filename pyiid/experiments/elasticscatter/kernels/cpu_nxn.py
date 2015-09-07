__author__ = 'christopher'
import math
from numba import *
import mkl

processor_target = 'cpu'


# F(Q) test_kernels -----------------------------------------------------------

@jit(target=processor_target)
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    n = len(q)
    for tx in range(n):
        for ty in range(n):
            for tz in range(3):
                d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


@jit(target=processor_target)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    n = len(r)
    for tx in range(n):
        for ty in range(n):
            r[tx, ty] = math.sqrt(
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@jit(target=processor_target)
def get_normalization_array(norm_array, scatter_array):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    n = len(norm_array)
    qmax_bin = norm_array.shape[2]

    for kq in range(0, qmax_bin):
        for tx in range(n):
            for ty in range(n):
                norm_array[tx, ty, kq] = (
                    scatter_array[tx, kq] * scatter_array[ty, kq])


@jit(target=processor_target)
def get_fq_array(fq, r, scatter_array, qbin):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    n = len(r)
    qmax_bin = len(fq)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    fq[kq] += scatter_array[tx, kq] * \
                              scatter_array[ty, kq] / \
                              r[tx, ty] * \
                              math.sin(kq * qbin * r[tx, ty])


@jit(target=processor_target, nopython=True)
def get_sigma_from_adp(sigma, adps, r, d):
    for i in xrange(len(sigma)):
        for j in xrange(len(sigma)):
            if i != j:
                tmp = 0.
                for w in range(3):
                    tmp += (math.fabs(adps[i, w]) + math.fabs(adps[j, w]))/2 \
                           * d[i, j, w] / r[i, j]
                sigma[i, j] = tmp ** 2


@jit(target=processor_target, nopython=True)
def get_dw_factor_from_sigma(dw_factor, sigma, qbin):
    for qx in xrange(dw_factor.shape[2]):
        Q = qx * qbin
        for i in xrange(len(sigma)):
            for j in xrange(len(sigma)):
                dw_factor[i, j, qx] = math.exp(-.5 * sigma[i, j] * Q ** 2)


@jit(target=processor_target, nopython=True)
def get_adp_fq(fq, r, norm, dw_factor, qbin):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    for qx in xrange(len(fq)):
        Q = float32(qbin) * float32(qx)
        for i in xrange(len(r)):
            for j in xrange(len(r)):
                if i != j:
                    fq[qx] += norm[i, j, qx] * dw_factor[i, j, qx] * \
                                   math.sin(Q * r[i, j]) / r[i, j]


# Gradient test_kernels -------------------------------------------------------
@jit(target=processor_target)
def fq_grad_position(grad_p, d, r, scatter_array, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    n = len(r)
    qmax_bin = grad_p.shape[2]
    for tx in range(n):
        for tz in range(3):
            for ty in range(n):
                if tx != ty:
                    for kq in range(0, qmax_bin):
                        sub_grad_p = \
                            scatter_array[tx, kq] * \
                            scatter_array[ty, kq] * \
                            d[tx, ty, tz] * \
                            (
                                (kq * qbin) *
                                r[tx, ty] *
                                math.cos(kq * qbin * r[tx, ty]) -
                                math.sin(kq * qbin * r[tx, ty])
                            ) \
                            / (r[tx, ty] ** 3)
                        grad_p[tx, tz, kq] += sub_grad_p
