from pyiid.wrappers import *

__author__ = 'christopher'
from threading import Thread

from pyiid.kernels.flat_kernel import get_ij_lists
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

    # setup flat map
    il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    jl = il.copy()
    get_ij_lists(il, jl, n)
    k_max = len(il)

    fq_q = []
    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            m = atoms_pdf_gpu_fq(n, qmax_bin, mem)
            if m > k_max - k_cov:
                m = k_max - k_cov
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if k_cov >= k_max:
                    break
                p = Thread(target=subs_fq, args=(
                    gpu, q, scatter_array, fq_q, qbin,
                    il[k_cov:k_cov + m],
                    jl[k_cov:k_cov + m],
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


def subs_grad_fq(gpu, q, scatter_array, grad_q, qbin, il, jl, k_cov,
                 index_list):
    # set up GPU
    with gpu:
        new_grad2 = atomic_grad_fq(q, scatter_array, qbin, il, jl)
    grad_q.append(new_grad2)
    index_list.append(k_cov)
    del k_cov, new_grad2


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem = setup_gpu_calc(atoms,
                                                                         sum_type)

    # setup flat map
    il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    jl = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    get_ij_lists(il, jl, n)
    # print il, jl
    k_max = len(il)

    gpus, mem_list = get_gpus_mem()
    grad_q = []
    index_list = []
    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            m = atoms_per_gpu_grad_fq(n, qmax_bin, mem)
            if m > k_max - k_cov:
                m = k_max - k_cov
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                p = Thread(target=subs_grad_fq, args=(
                    gpu, q, scatter_array, grad_q, qbin,
                    il[k_cov:k_cov + m],
                    jl[k_cov:k_cov + m],
                    k_cov,
                    index_list,
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m

                if k_cov >= k_max:
                    break
                    # TODO: sum arrays during processing to cut down on memory
    for value in p_dict.values():
        value.join()

    sort_grads = [x for (y, x) in sorted(zip(index_list, grad_q))]

    if len(sort_grads) > 1:
        # grads = np.concatenate(sort_grads, axis=)
        grad_p = np.sum(sort_grads, axis=0)
    else:
        grad_p = sort_grads[0]
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :] = np.nan_to_num(1 / na * grad_p[tx, tz, :])
    np.seterr(**old_settings)
    return grad_p
