from pyiid.experiments.elasticscatter.kernels import *
import math

__author__ = 'christopher'

processor_target = 'cpu'


# F(Q) test_kernels -----------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_d_array(d, q, offset):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    for k in range(len(d)):
        i, j = k_to_ij(i4(k + offset))
        for tz in range(3):
            d[k, tz] = q[i, tz] - q[j, tz]


@jit(target=processor_target, nopython=True)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    for k in xrange(len(r)):
        a, b, c = d[k, :]
        r[k] = math.sqrt(a * a + b * b + c * c)


@jit(target=processor_target, nopython=True)
def get_normalization_array(norm, scat, offset):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    for k in xrange(norm.shape[0]):
        i, j = k_to_ij(i4(k + offset))
        for qx in xrange(norm.shape[1]):
            norm[k, qx] = scat[i, qx] * scat[j, qx]


@jit(target=processor_target, nopython=True)
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
    kmax, qmax_bin = omega.shape
    for qx in xrange(qmax_bin):
        Q = f4(qbin) * f4(qx)
        for k in xrange(kmax):
            rk = r[k]
            omega[k, qx] = math.sin(Q * rk) / rk


@jit(target=processor_target, nopython=True)
def get_sigma_from_adp(sigma, adps, r, d, offset):
    for k in xrange(len(sigma)):
        i, j = k_to_ij(i4(k + offset))
        tmp = 0.
        for w in xrange(3):
            tmp += (adps[i, w] - adps[j, w]) * d[k, w] / r[k]
        sigma[k] = tmp


@jit(target=processor_target, nopython=True)
def get_tau(dw_factor, sigma, qbin):
    kmax, qmax_bin = dw_factor.shape
    for qx in xrange(qmax_bin):
        Q = f4(qbin) * f4(qx)
        for k in xrange(kmax):
            dw_factor[k, qx] = math.exp(f4(-.5) * f4(sigma[k] * sigma[k]) * f4(Q * Q))


@jit(target=processor_target, nopython=True)
def get_fq(fq, omega, norm):
    kmax, qmax_bin = omega.shape
    for qx in xrange(qmax_bin):
        for k in xrange(kmax):
            fq[k, qx] = norm[k, qx] * omega[k, qx]


@jit(target=processor_target, nopython=True)
def get_adp_fq(fq, omega, tau, norm):
    kmax, qmax_bin = omega.shape
    for qx in xrange(qmax_bin):
        for k in xrange(kmax):
            fq[k, qx] = norm[k, qx] * omega[k, qx] * tau[k, qx]


# Gradient test_kernels -------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_grad_omega(grad_omega, omega, r, d, qbin):
    kmax, _, qmax_bin = grad_omega.shape
    for qx in xrange(qmax_bin):
        Q = f4(qx) * f4(qbin)
        for k in xrange(kmax):
            rk = r[k]
            a = Q * math.cos(Q * rk) - omega[k, qx]
            a /= f4(rk * rk)
            for w in xrange(3):
                grad_omega[k, w, qx] = a * d[k, w]


@jit(target=processor_target, nopython=True)
def get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, offset):
    kmax, _, qmax_bin = grad_tau.shape
    for qx in xrange(qmax_bin):
        Q = qx * qbin
        for k in xrange(kmax):
            i, j = k_to_ij(k + offset)
            rk = r[k]
            tmp = f4(sigma[k] * f4(Q * Q) * tau[k, qx]) / f4(rk * rk * rk)
            for w in xrange(3):
                grad_tau[k, w, qx] = f4(tmp) * f4(
                    d[k, w] * sigma[k] - f4(adps[i, w] - adps[j, w]) * f4(rk * rk))


@jit(target=processor_target, nopython=True)
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
    kmax, _, qmax_bin = grad.shape
    for k in xrange(kmax):
        for w in xrange(3):
            for qx in xrange(qmax_bin):
                grad[k, w, qx] = norm[k, qx] * grad_omega[k, w, qx]


@jit(target=processor_target, nopython=True)
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
    kmax, _, qmax_bin = grad.shape
    for k in xrange(kmax):
        for w in xrange(3):
            for qx in xrange(qmax_bin):
                grad[k, w, qx] = norm[k, qx] * \
                                 (tau[k, qx] * grad_omega[k, w, qx] +
                                  omega[k, qx] * grad_tau[k, w, qx])


@jit(target=processor_target, nopython=True)
def fast_fast_flat_sum(new_grad, grad, k_cov):
    n = len(new_grad)
    k_max = len(grad)
    for i in xrange(n):
        for j in xrange(n):
            if j < i:
                k = j + i * (i - 1) / 2 - k_cov
                alpha = -1
            elif i < j:
                k = i + j * (j - 1) / 2 - k_cov
                alpha = 1
            else:
                k = -1
            if 0 <= k < k_max:
                for qx in xrange(grad.shape[2]):
                    for tz in xrange(3):
                        new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
