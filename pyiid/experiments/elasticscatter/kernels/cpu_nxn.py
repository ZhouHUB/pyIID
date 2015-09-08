import math
from numba import *
import mkl

__author__ = 'christopher'

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
    for i in xrange(n):
        for j in xrange(n):
            for w in xrange(3):
                d[i, j, w] = q[j, w] - q[i, w]


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
    for i in xrange(n):
        for j in xrange(n):
            r[i, j] = math.sqrt(
                d[i, j, 0] ** 2 + d[i, j, 1] ** 2 + d[i, j, 2] ** 2)


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
    for kq in xrange(norm_array.shape[2]):
        for i in xrange(n):
            for j in xrange(n):
                norm_array[i, j, kq] = scatter_array[i, kq] * \
                                       scatter_array[j, kq]


@jit(target=processor_target, nopython=True)
def get_sigma_from_adp(sigma, adps, r, d):
    for i in xrange(len(sigma)):
        for j in xrange(len(sigma)):
            if i != j:
                tmp = 0.
                for w in xrange(3):
                    tmp += (math.fabs(adps[i, w]) + math.fabs(adps[j, w])) / 2 \
                           * d[i, j, w] / r[i, j]
                sigma[i, j] = tmp ** 2


@jit(target=processor_target)
def get_omega(omega, r, qbin):
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
    for i in xrange(n):
        for j in xrange(n):
            if i != j:
                for kq in xrange(omega.shape[2]):
                    omega[i, j, kq] = math.sin(kq * qbin * r[i, j]) / r[i, j]


@jit(target=processor_target, nopython=True)
def get_tau(dw_factor, sigma, qbin):
    for qx in xrange(dw_factor.shape[2]):
        Q = qx * qbin
        for i in xrange(len(sigma)):
            for j in xrange(len(sigma)):
                dw_factor[i, j, qx] = math.exp(-.5 * sigma[i, j] * Q ** 2)


@jit(target=processor_target, nopython=True)
def get_fq(fq, omega, norm):
    for qx in xrange(len(fq)):
        for i in xrange(len(omega)):
            for j in xrange(len(omega)):
                fq[qx] += norm[i, j, qx] * omega[i, j, qx]


@jit(target=processor_target, nopython=True)
def get_adp_fq(fq, omega, tau, norm):
    for qx in xrange(len(fq)):
        for i in xrange(len(omega)):
            for j in xrange(len(omega)):
                fq[qx] += norm[i, j, qx] * omega[i, j, qx] * tau[i, j, qx]


# Gradient test_kernels -------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_grad_omega(grad_omega, omega, r, d, qbin):
    for qx in xrange(grad_omega.shape[3]):
        Q = qx * qbin
        for i in xrange(len(grad_omega)):
            for j in xrange(len(grad_omega)):
                if i != j:
                    rij = r[i, j]
                    A = Q * math.cos(Q * rij) - omega[i, j, qx]
                    A /= rij ** 2
                    for w in xrange(3):
                        grad_omega[i, j, w, qx] = A * d[i, j, w]


@jit(target=processor_target, nopython=True)
def get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin):
    for qx in xrange(grad_tau.shape[3]):
        Q = qx * qbin
        for i in xrange(len(grad_tau)):
            for j in xrange(len(grad_tau)):
                A = -1 * Q ** 2 * sigma[i, j] * tau[i, j, qx] / r[i, j] ** 3
                for w in xrange(3):
                    tmp = 0
                    for z in xrange(3):
                        tmp2 = 0
                        if z == w:
                            tmp2 = d[i, j, 0] ** 2 + d[i, j, 1] ** 2 + \
                                   d[i, j, 2] ** 2 - d[i, j, w] ** 2
                        else:
                            tmp2 = -1 * d[i, j, w] * d[i, j, z]
                        tmp += tmp2 * adps[i, j, z]
                    grad_tau[i, j, w, qx] = tmp * A


@jit(target=processor_target)
def get_grad_fq(grad, grad_omega, norm):
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
    n = len(grad)
    for i in xrange(n):
        for w in xrange(3):
            for j in xrange(n):
                if i != j:
                    for kq in xrange(grad.shape[2]):
                        grad[i, w, kq] += norm[i, j, kq] * grad_omega[i, j, w,
                                                                      kq]


@jit(target=processor_target)
def get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm):
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
    n = len(grad)
    qmax_bin = grad.shape[3]
    for i in xrange(n):
        for w in xrange(3):
            for j in xrange(n):
                if i != j:
                    for kq in xrange(qmax_bin):
                        grad[i, w, kq] += norm[i, j, kq] * \
                                          (tau[i, j, kq] *
                                           grad_omega[i, j, w, kq] +
                                           omega[i, j, kq] *
                                           grad_tau[i, j, w, kq])
