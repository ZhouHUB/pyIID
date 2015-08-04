import math
from numbapro.cudalib import cufft
import numpy as np
from pyiid.wrappers import *
from pyiid.wrappers import get_gpus_mem

__author__ = 'christopher'
from threading import Thread
from pyiid.wrappers.k_atomic_gpu import *


def subs_fq(gpu, q, scatter_array, fq_q, qbin, il, jl):
    # set up GPU
    with gpu:
        final = atomic_fq(q, scatter_array, qbin, il, jl)
        fq_q.append(final)
        del final


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, gpus, mem_list = setup_gpu_calc(atoms,
                                                                   sum_type)

    # setup test_flat map
    # il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    # jl = il.copy()
    # get_ij_lists(il, jl, n)
    k_max = int((n ** 2 - n) / 2.)

    fq_q = []
    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                m = atoms_pdf_gpu_fq(n, qmax_bin, mem)
                if m > k_max - k_cov:
                    m = k_max - k_cov
                if k_cov >= k_max:
                    break
                p = Thread(target=subs_fq, args=(
                    gpu, q, scatter_array, fq_q, qbin, m, k_cov
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m

                if k_cov >= k_max:
                    break
    for value in p_dict.values():
        value.join()
    final = np.zeros(qmax_bin)
    for ele in fq_q:
        final[:] += ele
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    return 2 * final


def subs_grad_fq(gpu, q, scatter_array, grad_q, qbin, k_cov, m):
    # set up GPU
    with gpu:
        new_grad2 = atomic_grad_fq(q, scatter_array, qbin, k_cov, m)
    grad_q.append(new_grad2)
    del k_cov, new_grad2


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem = setup_gpu_calc(atoms,
                                                                         sum_type)

    # setup test_flat map
    k_max = int((n ** 2 - n) / 2.)

    gpus, mem_list = get_gpus_mem()
    grad_q = []
    k_cov = 0
    p_dict = {}
    grad_p = np.zeros((n, 3, qmax_bin))
    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                m = atoms_per_gpu_grad_fq(n, qmax_bin, mem)
                if m > k_max - k_cov:
                    m = k_max - k_cov

                p = Thread(target=subs_grad_fq, args=(
                    gpu, q, scatter_array, grad_q, qbin, k_cov, m,
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m
                # print float(k_cov) / k_max * 100., '%'
                if k_cov >= k_max:
                    break
        # TODO: sum arrays during processing to cut down on memory
        # if queue is not empty sum the queue down
        # remove entries from queue
        for i in range(len(grad_q)):
            # print len(grad_q)
            grad_p += grad_q.pop(0)

    for value in p_dict.values():
        value.join()
    # print len(grad_q)
    for i in range(len(grad_q)):
        grad_p += grad_q.pop(0)
    # print len(grad_q)
    # grad_p = np.sum(grad_q, axis=0)
    # '''
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :] = np.nan_to_num(1 / na * grad_p[tx, tz, :])
    np.seterr(**old_settings)
    # '''
    return grad_p


def sub_grad_pdf(gpu, input_shape, gpadc, gpadcfft, j, n_cov):
    with gpu:
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

    gpus, mems = get_gpus_mem()
    n_cov = 0
    p_dict = {}

    while n_cov < n:
        for gpu, mem in zip(gpus, mems):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                j = int(math.floor(mem / gpadcfft.shape[-1] / 64 / 2))
                if j > n - n_cov:
                    j = n - n_cov
                if n_cov >= n:
                    break
                p = Thread(target=sub_grad_pdf, args=(gpu, input_shape, gpadc, gpadcfft, j, n_cov))
                p.start()
                p_dict[gpu] = p
                n_cov += j
                if n_cov >= n:
                    break
    for value in p_dict.values():
        value.join()
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