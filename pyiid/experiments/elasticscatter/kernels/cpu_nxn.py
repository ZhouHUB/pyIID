import math
from numba import *
import mkl
import os

__author__ = 'christopher'

cache = True
if bool(os.getenv('NUMBA_DISABLE_JIT')):
    cache = False
processor_target = 'cpu'


# F(sv) test_kernels ----------------------------------------------------------

@jit(void(f4[:, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
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
    for i in xrange(i4(n)):
        for j in xrange(i4(n)):
            for w in xrange(i4(3)):
                d[i, j, w] = q[j, w] - q[i, w]


@jit(void(f4[:, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
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
    for i in xrange(i4(n)):
        for j in xrange(i4(n)):
            tmp = f4(0.)
            for w in xrange(i4(3)):
                tmp += d[i, j, w] * d[i, j, w]
            r[i, j] = math.sqrt(tmp)


@jit(void(f4[:, :, :], f4[:, :]), target=processor_target, nopython=True,
     cache=cache)
def get_normalization_array(norm_array, scatter_array):
    """
    Generate the sv dependant normalization factors for the F(sv) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    n, _, qmax_bin = norm_array.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    norm_array[i, j, qx] = scatter_array[i, qx] * \
                                           scatter_array[j, qx]


@jit(void(f4[:, :], f4[:, :], f4[:, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_sigma_from_adp(sigma, adps, r, d):
    for i in xrange(i4(len(sigma))):
        for j in xrange(i4(len(sigma))):
            if i != j:
                tmp = f4(0.)
                for w in xrange(i4(3)):
                    tmp += (adps[i, w] - adps[j, w]) * d[i, j, w] / r[i, j]
                sigma[i, j] = tmp


@jit(void(f4[:, :, :], f4[:, :], f4), target=processor_target, nopython=True,
     cache=cache)
def get_omega(omega, r, qbin):
    """
    Generate F(sv), not normalized, via the Debye sum

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
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    rij = r[i, j]
                    omega[i, j, qx] = math.sin(sv * rij) / rij


@jit(void(f4[:, :, :], f4[:, :], f4), target=processor_target, nopython=True,
     cache=cache)
def get_tau(dw_factor, sigma, qbin):
    n, _, qmax_bin = dw_factor.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                dw_factor[i, j, qx] = math.exp(
                    f4(-.5) * sigma[i, j] * sigma[i, j] * sv * sv)


@jit(void(f4[:], f4[:, :, :], f4[:, :, :]), target=processor_target,
     nopython=True, cache=cache)
def get_fq(fq, omega, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                fq[qx] += norm[i, j, qx] * omega[i, j, qx]


@jit(void(f4[:, :, :], f4[:, :, :], f4[:, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_adp_fq(fq, omega, tau, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                fq[i, j, qx] = norm[i, j, qx] * omega[i, j, qx] * tau[i, j, qx]


@jit(void(f4[:, :, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
def get_fq_inplace(omega, norm):
    n, _, qmax_bin = omega.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                omega[i, j, qx] *= norm[i, j, qx]


# Gradient test_kernels -------------------------------------------------------


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :], f4[:, :, :], f4),
     target=processor_target, nopython=True, cache=cache)
def get_grad_omega(grad_omega, omega, r, d, qbin):
    n, _, _, qmax_bin = grad_omega.shape
    for qx in xrange(i4(qmax_bin)):
        sv = f4(qx) * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    rij = r[i, j]
                    a = sv * math.cos(sv * rij) - omega[i, j, qx]
                    a /= rij * rij
                    for w in xrange(i4(3)):
                        grad_omega[i, j, w, qx] = a * d[i, j, w]


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :], f4[:, :, :], f4[:, :],
          f4[:, :, ], f4), target=processor_target, nopython=True, cache=cache)
def get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin):
    n, _, _, qmax_bin = grad_tau.shape
    for qx in xrange(i4(qmax_bin)):
        sv = qx * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    tmp = sigma[i, j] * sv ** 2 * tau[i, j, qx] / r[i, j] ** 3
                    for w in xrange(i4(3)):
                        grad_tau[i, j, w, qx] = tmp * (
                            d[i, j, w] * sigma[i, j] -
                            (adps[i, w] - adps[j, w]) *
                            r[i, j] ** 2)


@jit(void(f4[:, :, :, :], f4[:, :, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_grad_fq(grad, grad_omega, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

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
        The size of the sv bins
    """
    n, _, _, qmax_bin = grad.shape
    for i in xrange(i4(n)):
        for w in xrange(i4(3)):
            for j in xrange(i4(n)):
                if i != j:
                    for qx in xrange(grad.shape[3]):
                        grad[i, j, w, qx] = norm[i, j, qx] * grad_omega[
                            i, j, w, qx]


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :, :], f4[:, :, :, :],
          f4[:, :, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

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
        The size of the sv bins
    """
    n, _, _, qmax_bin = grad.shape
    for i in xrange(i4(n)):
        for w in xrange(i4(3)):
            for j in xrange(i4(n)):
                if i != j:
                    for qx in xrange(i4(qmax_bin)):
                        grad[i, j, w, qx] = norm[i, j, qx] * \
                                            (tau[i, j, qx] *
                                             grad_omega[i, j, w, qx] +
                                             omega[i, j, qx] *
                                             grad_tau[i, j, w, qx])


@jit(void(f4[:, :, :, :], f4[:, :, :]), target=processor_target, nopython=True,
     cache=cache)
def get_grad_fq_inplace(grad_omega, norm):
    """
    Generate the gradient F(sv) for an atomic configuration

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
        The size of the sv bins
    """
    n, _, _, qmax_bin = grad_omega.shape
    for i in xrange(i4(n)):
        for w in xrange(i4(3)):
            for j in xrange(i4(n)):
                if i != j:
                    for qx in xrange(i4(qmax_bin)):
                        grad_omega[i, j, w, qx] *= norm[i, j, qx]


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :], f4[:, :], f4[:, :, :], f4),
     target=processor_target, nopython=True, cache=cache)
def get_dtau_dadp(dtau_dadp, tau, sigma, r, d, qbin):
    n, _, _, qmax_bin = dtau_dadp.shape
    for qx in xrange(i4(qmax_bin)):
        sv = qx * qbin
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                if i != j:
                    tmp = f4(-1.) * sigma[i, j] * sv * sv * tau[i, j, qx] / r[
                        i, j]
                    for w in xrange(i4(3)):
                        dtau_dadp[i, j, w, qx] = tmp * d[i, j, w]


@jit(void(f4[:, :, :, :], f4[:, :, :], f4[:, :, :]),
     target=processor_target, nopython=True, cache=cache)
def get_dfq_dadp_inplace(dtau_dadp, omega, norm):
    n, _, _, qmax_bin = dtau_dadp.shape
    for qx in xrange(i4(qmax_bin)):
        for i in xrange(i4(n)):
            for j in xrange(i4(n)):
                for w in xrange(i4(3)):
                    dtau_dadp[i, j, w, qx] *= norm[i, j, qx] * omega[i, j, qx]


@jit(target=processor_target, nopython=True, cache=cache)
def lerp(t, a, b):
    x = (f4(1) - t) * a
    y = b * t
    return x + y


@jit(void(f4[:, :, :], f4[:, :], f4[:], f4),
     target=processor_target, nopython=True, cache=cache)
def get_3d_overlap(voxels, q, pdf, rstep):
    n, _ = q.shape
    im, jm, km = voxels.shape
    for i in xrange(im):
        for j in xrange(jm):
            for k in xrange(km):
                tmp = f4(0.)
                ai = (f4(i) + f4(.5)) * rstep
                bj = (f4(j) + f4(.5)) * rstep
                ck = (f4(k) + f4(.5)) * rstep
                for l in xrange(n):
                    a = ai - q[l, 0]
                    b = bj - q[l, 1]
                    c = ck - q[l, 2]
                    d = a * a + b * b + c * c
                    r = math.sqrt(d)
                    tmp += lerp((r / rstep - math.floor(r / rstep)),
                                pdf[i4(math.floor(r / rstep))],
                                pdf[i4(math.ceil(r / rstep))])
                voxels[i, j, k] += tmp


@jit(void(f4[:, :, :], f4[:, :], f4, f4),
     target=processor_target, nopython=True, cache=cache)
def zero_occupied_atoms(voxels, q, rstep, rmin):
    n, _ = q.shape
    im, jm, km = voxels.shape
    for i in xrange(im):
        for j in xrange(jm):
            for k in xrange(km):
                for l in xrange(n):
                    a = (f4(i) + f4(.5)) * rstep - q[l, 0]
                    b = (f4(j) + f4(.5)) * rstep - q[l, 1]
                    c = (f4(k) + f4(.5)) * rstep - q[l, 2]
                    d = a * a + b * b + c * c
                    r = math.sqrt(d)
                    if r <= rmin:
                        voxels[i, j, k] = f4(0.)


@jit(void(f4[:, :], f4[:, :], f4[:], f4),
     target=processor_target, nopython=True, cache=cache)
def get_atomic_overlap(voxels, r, pdf, rstep):
    im, jm = r.shape
    for i in xrange(im):
        for j in xrange(jm):
            rk = r[i, j]
            voxels[i, j] += lerp(rk / rstep - math.floor(rk / rstep),
                                 pdf[i4(math.floor(rk / rstep))],
                                 pdf[i4(math.ceil(rk / rstep))])
