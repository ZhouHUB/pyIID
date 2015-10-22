from threading import Thread

import numpy as np

from pyiid.wrappers import *
from pyiid.wrappers.gpu_wrappers.nxn_atomic_gpu import atoms_per_gpu_fq, \
    atoms_per_gpu_grad_fq
__author__ = 'christopher'

print 'Warning: this module is no longer fully supported/test, ' \
      'and may not be complete or robust. Please use the flat_gpu_wrap module'


def sub_fq(gpu, q, scatter_array, fq_q, qbin, m, n_cov):
    with gpu:
        from pyiid.wrappers.gpu_wrappers.nxn_atomic_gpu import atomic_fq

        final = atomic_fq(q, scatter_array, qbin, m, n_cov)
        fq_q.append(final)
        del final


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem = setup_gpu_calc(
        atoms, sum_type)
    # starting buffers
    fq_q = []
    n_cov = 0
    p_dict = {}

    # The total amount of work is greater than the sum of our GPUs, no
    # special distribution needed, just keep putting problems on GPUs until
    # finished.
    while n_cov < n:
        for gpu, mem in zip(sort_gpus, sort_gmem):
            m = atoms_per_gpu_fq(n, qmax_bin, mem)
            if m > n - n_cov:
                m = n - n_cov

            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if n_cov >= n:
                    break
                p = Thread(
                    target=sub_fq, args=(
                        gpu, q, scatter_array,
                        fq_q, qbin, m, n_cov))
                p.start()
                p_dict[gpu] = p
                n_cov += m

                if n_cov >= n:
                    break

    for value in p_dict.values():
        value.join()
    fq = np.zeros(qmax_bin)
    for ele in fq_q:
        fq[:] += ele
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def sub_grad(gpu, q, scatter_array, grad_q, qbin, m, n_cov, index_list):
    with gpu:
        from pyiid.wrappers.gpu_wrappers.nxn_atomic_gpu import atomic_grad_fq

        final, _ = atomic_grad_fq(q, scatter_array, qbin, m, n_cov)
        grad_q.append(final)
        index_list.append(n_cov)


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem = setup_gpu_calc(
        atoms, sum_type)

    grad_q = []
    index_list = []
    p_dict = {}
    n_cov = 0
    while n_cov < n:
        for gpu, mem in zip(sort_gpus, sort_gmem):
            m = atoms_per_gpu_grad_fq(n, qmax_bin, mem)
            if m > n - n_cov:
                m = n - n_cov
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                p = Thread(
                    target=sub_grad, args=(
                        gpu, q, scatter_array,
                        grad_q, qbin, m, n_cov, index_list))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov == n:
                    break
    for value in p_dict.values():
        value.join()

    # Sort grads to make certain indices are in order
    sort_grads = [x for (y, x) in sorted(zip(index_list, grad_q))]

    # len(sort_grads)
    # sorted_sum_grads = [x.sum(axis=(1)) for x in sort_grads]

    # Stitch arrays together
    if len(sort_grads) > 1:
        grad_p = np.concatenate(sort_grads, axis=0)
    else:
        grad_p = sort_grads[0]

    # sum reduce to 1D
    na = np.average(scatter_array, axis=0) ** 2 * n

    old_settings = np.seterr(all='ignore')
    grad_p[:, :] = np.nan_to_num(grad_p[:, :] / na)
    np.seterr(**old_settings)
    return grad_p
