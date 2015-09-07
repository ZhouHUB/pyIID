__author__ = 'christopher'

from pyiid.experiments.elasticscatter.kernels import *

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
        i, j = k_to_ij(k)
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
        r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)


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
        i, j = k_to_ij(k)
        for qx in xrange(norm.shape[1]):
            norm[k, qx] = scat[i, qx] * scat[j, qx]


@jit(target=processor_target, nopython=True)
def get_fq(fq, r, norm, qbin):
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
    for qx in xrange(fq.shape[1]):
        Q = float32(qbin) * float32(qx)
        for k in xrange(fq.shape[0]):
            rk = r[k]
            fq[k, qx] = norm[k, qx] * math.sin(Q * rk) / rk


@jit(target=processor_target, nopython=True)
def get_sigma_from_adp(sigma, adps, r, d):
    for k in xrange(len(sigma)):
        i, j = k_to_ij(k)
        tmp = 0.
        for w in range(3):
            tmp += (adps[i, w] - adps[j, w]) * d[i, j, w] / r[k]
        sigma[k] = tmp ** 2


@jit(target=processor_target, nopython=True)
def get_dw_factor_from_sigma(dw_factor, sigma, qbin):
    for qx in xrange(dw_factor.shape[1]):
        Q = qx * qbin
        for k in xrange(len(sigma)):
            dw_factor[k, qx] = math.exp(-.5 * sigma[k] * Q ** 2)


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
    for qx in xrange(fq.shape[1]):
        Q = float32(qbin) * float32(qx)
        for k in xrange(fq.shape[0]):
            rk = r[k]
            fq[k, qx] = norm[k, qx] * dw_factor[k, qx] * \
                        math.sin(Q * rk) / rk


# Gradient test_kernels -------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_grad_fq(grad, fq, r, d, norm, qbin):
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
    for qx in xrange(fq.shape[1]):
        Q = float32(qbin) * float32(qx)
        for k in xrange(grad.shape[0]):
            rk = r[k]
            A = (norm[k, qx] * Q * math.cos(Q * rk) - fq[k, qx]) / \
                float32(rk * rk)
            for w in range(3):
                grad[k, w, qx] = A * d[k, w]


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
            if 0 <= k < k_max:
                for qx in xrange(grad.shape[2]):
                    for tz in xrange(3):
                        new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
