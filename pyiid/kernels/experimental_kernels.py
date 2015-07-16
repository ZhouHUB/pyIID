__author__ = 'christopher'
import math

from numba import *
import numpy as np
from numbapro.cudalib import cufft
import matplotlib.pyplot as plt
from threading import Thread
from pyiid.wrappers import get_gpus_mem


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


# def fft_gr_to_fq(g, rstep, rmin):
#     """
#     Fourier Transform from G(r) to F(Q)
#
#     Parameters
#     ----------
#     :param rmin:
#     g: Nd array
#         The PDF
#     rbin: float
#         The size of the distance bins
#
#     Returns
#     -------
#     f: Nd array
#         The reduced structure factor
#     """
#     if g is None: return g
#     padrmin = int(round(rmin / rstep))
#     npad1 = padrmin + len(g)
#
#     # pad to the next power of 2 for fast Fourier transformation
#     npad2 = (1 << int(math.ceil(math.log(npad1, 2)))) * 2
#     # sine transformations needs an odd extension
#
#     npad4 = 4 * npad2
#     # gpadc array has to be doubled for complex coefficients
#     gpadc = np.zeros(npad4)
#     gpadc2 = np.zeros(npad4)
#     # copy the original g signal
#     ilo = 0
#     # ilo = padrmin
#     # ilo = len(g)
#     gpadc[:2 * len(g):2] = g[:]
#     # gpadc2[:2 * len(g):2] = g[:]
#     '''
#     for i in xrange(len(g)):
#         gpadc[2 * ilo] = g[i]
#         ilo += 1
#
#     assert_allclose(gpadc2, gpadc)
#     # '''
#     # copy the odd part of g skipping the first point,
#     # because it is periodic image of gpadc[0]
#     gpadc[-2:-2 * len(g) + 1:-2] = -1 * g[1:]
#     # gpadc2[-2:-2 * len(g) + 1:-2] = -1 * g[1:]
#     '''
#     ihi = 2 * npad2 - 1
#     for ilo in xrange(1, npad2):
#         gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
#         ihi -= 1
#
#     assert_allclose(gpadc2, gpadc)
#     # '''
#     # plt.plot(gpadc)
#     # plt.show()
#
#     # gpadcfft = np.fft.ihfft(gpadc)
#     gpadcfft = np.fft.ifft(gpadc)
#     # plt.plot(gpadcfft.imag)
#     # plt.show()
#
#     f = np.zeros(npad2, dtype=complex)
#     # f2 = np.zeros(npad2, dtype=complex)
#     f[:] = gpadcfft[:npad2 * 2:2] * npad2 * rstep
#     # f2[:] = gpadcfft[:npad2 * 2:2] * npad2 * rstep
#     '''
#     for i in xrange(npad2):
#         # f[i] = gpadcfft[2 * i + 1] * npad2 * rstep
#         f[i] = gpadcfft[2 * i] * npad2 * rstep
#     assert_allclose(f2, f)
#     # '''
#     return f.imag
#
#
# def fft_gr_to_fq(g, rstep, rmin):
#     """
#     Fourier Transform from G(r) to F(Q)
#
#     Parameters
#     ----------
#     :param rmin:
#     g: Nd array
#         The PDF
#     rbin: float
#         The size of the distance bins
#
#     Returns
#     -------
#     f: Nd array
#         The reduced structure factor
#     """
#     print 'hi'
#     if g is None: return g
#     padrmin = int(round(rmin / rstep))
#     npad1 = padrmin + len(g)
#
#     # pad to the next power of 2 for fast Fourier transformation
#     npad2 = (1 << int(math.ceil(math.log(npad1, 2)))) * 2
#     # sine transformations needs an odd extension
#
#     npad4 = 4 * npad2
#     # gpadc array has to be doubled for complex coefficients
#     gpadc = np.zeros(npad4)
#     # copy the original g signal
#     ilo = 0
#     # ilo = padrmin
#     for i in range(len(g)):
#         gpadc[2 * ilo] = g[i]
#         ilo += 1
#     # copy the odd part of g skipping the first point,
#     # because it is periodic image of gpadc[0]
#     ihi = 2 * npad2 - 1
#     for ilo in range(1, npad2):
#         gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
#         ihi -= 1
#
#     # plt.plot(gpadc)
#     # plt.show()
#
#     gpadcfft2 = np.fft.ifft(gpadc)
#
#     cufft.FFTPlan(shape=gpadc.shape, itype=np.complex64, otype=np.complex64)
#     gpadc = gpadc.astype(np.complex64)
#     dgpadc = cuda.to_device(gpadc)
#     gpadcfft = np.zeros(gpadc.shape, np.complex64)
#     dgpadcfft = cuda.to_device(gpadcfft)
#     cufft.ifft(dgpadc, dgpadcfft)
#     dgpadcfft.to_host()
#     gpadcfft /= len(gpadcfft)
#     from numpy.testing import assert_allclose
#
#     plt.plot(gpadcfft.imag - gpadcfft2.imag)
#     plt.show()
#     assert_allclose(gpadcfft.imag, gpadcfft2.imag, atol=1e-8)
#     f = np.zeros(npad2, dtype=complex)
#     for i in range(npad2):
#         # f[i] = gpadcfft[2 * i + 1] * npad2 * rstep
#         f[i] = gpadcfft[2 * i] * npad2 * rstep
#     return f.imag
#
#
# # @autojit(target=targ)
# def get_pdf_at_qmin(fpad, rstep, qstep, rgrid, qmin):
#     """
#     Get the atomic pair distribution function
#
#     Parameters
#     -----------
#     :param qmin:
#     fpad: 1d array
#         The reduced structure function, padded with zeros to qmin
#     rstep: float
#         The step size in real space
#     qstep: float
#         The step size in inverse space
#     rgrid: 1d array
#         The real space r values
#     rmax: float
#         The maximum r value
#
#     Returns
#     -------
#     1d array:
#         The atomic pair distribution function
#     """
#     # Zero out F(Q) below qmin theshold
#     fpad[:int(math.ceil(qmin / qstep))] = 0.0
#     # Expand F(Q)
#     nfromdr = int(math.ceil(math.pi / rstep / qstep))
#     if nfromdr > int(len(fpad)):
#         # put in a bunch of zeros
#         fpad2 = np.zeros(nfromdr)
#         fpad2[:len(fpad)] = fpad
#         fpad = fpad2
#
#     gpad = fft_fq_to_gr(fpad, qstep, qmin)
#
#     drpad = math.pi / (len(gpad) * qstep)
#     pdf0 = np.zeros(len(rgrid))
#     for i, r in enumerate(rgrid):
#         xdrp = r / drpad / 2
#         # xdrp = r / drpad
#         iplo = int(xdrp)
#         iphi = iplo + 1
#         wphi = xdrp - iplo
#         wplo = 1.0 - wphi
#         pdf0[i] = wplo * gpad[iplo] + wphi * gpad[iphi]
#     pdf1 = pdf0 * 2
#     return pdf1.real
#     # return gpad
#
#
# # @autojit(target='cpu')
# def fft_fq_to_gr(f, qbin, qmin):
#     """
#     Fourier Transform from F(Q) to G(r)
#
#     Parameters
#     -----------
#     :param qmin:
#     f: Nd array
#         F(Q)
#     qbin: float
#         Qbin size
#     qmin:
#
#     Returns
#     -------
#     g: Nd array
#         The PDF
#     """
#     g = fft_gr_to_fq(f, qbin, qmin)
#     g *= 2.0 / math.pi
#     return g


def get_pdf_at_qmin(fpad_array, rstep, qstep, rgrid, qmin):
    n = len(fpad_array)
    fpad_array[:, int(math.ceil(qmin / qstep))] = 0.0
    # Expand F(Q)
    nfromdr = int(math.ceil(math.pi / rstep / qstep))
    if nfromdr > int(fpad_array.shape[-1]):
        # put in a bunch of zeros
        fpad2 = np.zeros((n, nfromdr))
        fpad2[:, :fpad_array.shape[-1]] = fpad_array[:, :]
        fpad_array = fpad2

    batch_input = np.ravel(fpad_array).astype(np.complex64)
    input_shape = (fpad_array[0].shape)
    batch_operations = n
    plan = cufft.FFTPlan(input_shape, batch_input.dtype, np.complex64,
                         batch_operations)
    batch_output = np.zeros(batch_input.shape, dtype=np.complex64)
    _ = plan.inverse(batch_input, out=batch_output)

    data_out = np.reshape(batch_output, (n, input_shape[0]))
    fft_len = input_shape

    data_out = data_out.imag
    data_out /= fft_len
    data_out *= 2.0 / math.pi
    drpad = math.pi / (data_out.shape[-1] * qstep)

    axdrp = rgrid / drpad / 2
    aiplo = axdrp.astype(np.int)
    aiphi = aiplo + 1
    awphi = axdrp - aiplo
    awplo = 1.0 - awphi
    # pdf0a[:] = awplo[:] * data_out[aiplo] + awphi * data_out[aiphi]
    pdf0 = awplo[:] * data_out[:, aiplo] + awphi[:] * data_out[:, aiphi]
    pdf1 = pdf0 * 2
    return pdf1


def grad_pdf(grad_fq, rstep, qstep, rgrid, qmin):
    gpus, mem = get_gpus_mem()
    pdf_grad = np.zeros((len(grad_fq), 3, len(rgrid)))
    for i in range(3):
        pdf_grad[:, i, :] = get_pdf_at_qmin(grad_fq[:, i, :], rstep, qstep, rgrid, qmin)
    return pdf_grad

'''
@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def fast_fast_flat_sum(new_grad, grad, k_cov):
    i, j, qx = cuda.grid(3)
    # i, qx = cuda.grid(2)
    n = len(new_grad)
    k_max = len(grad)
    if i >= n or j >= n or qx >= grad.shape[2]:
        return
    # if i >= n or qx >= grad.shape[2]:
    #     return
    for tz in xrange(3):
        tmp = float32(0.0)
        # for j in xrange(n):
        if j < i:
            k = cuda_ij_to_k(i, j) - k_cov
            alpha = float32(-1)
        elif i < j:
            k = cuda_ij_to_k(j, i) - k_cov
            alpha = float32(1)
        if 0 <= k < k_max:
            # new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
            tmp += grad[k, tz, qx] * alpha
        new_grad[i, tz, qx] += tmp
'''