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


def grad_pdf(fpad, rstep, qstep, rgrid, qmin):
    n = len(fpad)
    fpad[:, :, :int(math.ceil(qmin / qstep))] = 0.0
    # Expand F(Q)
    nfromdr = int(math.ceil(math.pi / rstep / qstep))
    if nfromdr > int(len(fpad)):
        # put in a bunch of zeros
        fpad2 = np.zeros((n, 3, nfromdr))
        fpad2[:, :, :fpad.shape[-1]] = fpad
        fpad = fpad2

    padrmin = int(round(qmin / qstep))
    npad1 = padrmin + fpad.shape[-1]

    npad2 = (1 << int(math.ceil(math.log(npad1, 2)))) * 2

    npad4 = 4 * npad2
    gpadc = np.zeros((n, 3, npad4))
    gpadc[:, :, :2 * fpad.shape[-1]:2] = fpad[:, :, :]
    gpadc[:, :, -2:-2 * fpad.shape[-1] + 1:-2] = -1 * fpad[:, :, 1:]
    gpadcfft = np.zeros(gpadc.shape, dtype=complex)

    input_shape = [gpadcfft.shape[-1]]

    mem = cuda.current_context().get_memory_info()[0]
    n_cov = 0
    while n_cov < n:
        j = int(math.floor(mem / gpadcfft.shape[-1] / 64 / 2))
        if j > n - n_cov:
            j = n - n_cov
        batch_operations = j
        plan = cufft.FFTPlan(input_shape, np.complex64, np.complex64,
                                 batch_operations)
        for i in range(3):
            batch_input = np.ravel(gpadc[n_cov:n_cov + j, i, :]).astype(np.complex64)
            batch_output = np.zeros(batch_input.shape, dtype=np.complex64)

            _ = plan.inverse(batch_input, out=batch_output)
            del batch_input
            data_out = np.reshape(batch_output, (j, input_shape[0]))
            data_out /= input_shape[0]

            gpadcfft[n_cov:n_cov + j, i, :] = data_out
            del data_out, batch_output
        n_cov += j

    g = np.zeros((n, 3, npad2), dtype=complex)
    g[:, :, :] = gpadcfft[:, :, :npad2 * 2:2] * npad2 * qstep

    gpad = g.imag * 2.0 / math.pi
    drpad = math.pi / (gpad.shape[-1] * qstep)

    pdf0 = np.zeros((n, 3, len(rgrid)))
    axdrp = rgrid / drpad / 2
    aiplo = axdrp.astype(np.int)
    aiphi = aiplo + 1
    awphi = axdrp - aiplo
    awplo = 1.0 - awphi
    pdf0[:, :, :] = awplo[:] * gpad[:, :, aiplo] + awphi * gpad[:, :, aiphi]
    pdf1 = pdf0 * 2
    return pdf1.real

# def grad_pdf(grad_fq, rstep, qstep, rgrid, qmin):
#     pdf_grad = get_pdf_at_qmin(grad_fq, rstep, qstep, rgrid, qmin)
#     return pdf_grad

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
