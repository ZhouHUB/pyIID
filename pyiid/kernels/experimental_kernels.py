__author__ = 'christopher'
from numba import *
import math
from pyiid.kernels.flat_kernel import cuda_ij_to_k, cuda_k_to_ij
import numpy as np
from numbapro.cudalib import cufft
import matplotlib.pyplot as plt

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
def fast_fast_flat_sum(new_grad, grad, k_cov):
    i, j, qx = cuda.grid(3)
    n = len(new_grad)
    k_max = len(grad)
    if i == j or i >= n or j >= n or qx >= grad.shape[2]:
        return
    for tz in xrange(3):
        if j < i:
            k = j + i * (i - 1) / 2 - k_cov
            alpha = -1
        elif i < j:
            k = i + j * (j - 1) / 2 - k_cov
            alpha = 1
        if 0 <= k < k_max:
            new_grad[i, tz, qx] += grad[k, tz, qx] * alpha

def fft_gr_to_fq(g, rstep, rmin):
    """
    Fourier Transform from G(r) to F(Q)

    Parameters
    ----------
    :param rmin:
    g: Nd array
        The PDF
    rbin: float
        The size of the distance bins

    Returns
    -------
    f: Nd array
        The reduced structure factor
    """
    print 'hi'
    if g is None: return g
    padrmin = int(round(rmin / rstep))
    npad1 = padrmin + len(g)

    # pad to the next power of 2 for fast Fourier transformation
    npad2 = (1 << int(math.ceil(math.log(npad1, 2)))) * 2
    # sine transformations needs an odd extension

    npad4 = 4 * npad2
    # gpadc array has to be doubled for complex coefficients
    gpadc = np.zeros(npad4)
    # copy the original g signal
    ilo = 0
    # ilo = padrmin
    for i in range(len(g)):
        gpadc[2 * ilo] = g[i]
        ilo += 1
    # copy the odd part of g skipping the first point,
    # because it is periodic image of gpadc[0]
    ihi = 2 * npad2 - 1
    for ilo in range(1, npad2):
        gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
        ihi -= 1

    # plt.plot(gpadc)
    # plt.show()

    gpadcfft2 = np.fft.ifft(gpadc)

    cufft.FFTPlan(shape=gpadc.shape, itype=np.complex64, otype=np.complex64)
    gpadc = gpadc.astype(np.complex64)
    dgpadc = cuda.to_device(gpadc)
    gpadcfft = np.zeros(gpadc.shape, np.complex64)
    dgpadcfft = cuda.to_device(gpadcfft)
    cufft.ifft(dgpadc, dgpadcfft)
    dgpadcfft.to_host()
    gpadcfft /= len(gpadcfft)
    from numpy.testing import assert_allclose


    plt.plot(gpadcfft.imag - gpadcfft2.imag)
    plt.show()
    assert_allclose(gpadcfft.imag, gpadcfft2.imag, atol=1e-8)
    f = np.zeros(npad2, dtype=complex)
    for i in range(npad2):
        # f[i] = gpadcfft[2 * i + 1] * npad2 * rstep
        f[i] = gpadcfft[2 * i] * npad2 * rstep
    return f.imag