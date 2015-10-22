from multiprocessing import Pool, cpu_count

import numpy as np
import psutil

from pyiid.kernels.cpu_flat import *
from pyiid.wrappers.gpu_wrappers.k_atomic_gpu import atoms_pdf_gpu_fq, atoms_per_gpu_grad_fq
__author__ = 'christopher'


def setup_gpu_calc(atoms, sum_type):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter').astype(np.float32)
    else:
        scatter_array = atoms.get_array('PDF scatter').astype(np.float32)
    qmax_bin = scatter_array.shape[1]

    return q, n, qmax_bin, scatter_array, None, None


def atomic_fq(task):
    q, scatter_array, qbin, k_max, k_cov = task
    qmax_bin = scatter_array.shape[1]

    d = np.zeros((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)
    del q

    r = np.zeros(k_max, np.float32)
    get_r_array(r, d)
    del d

    norm = np.zeros((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)
    del scatter_array

    fq = np.zeros((k_max, qmax_bin), np.float32)
    get_fq(fq, r, norm, qbin)
    del norm, r

    rtn = fq.sum(axis=0)
    del fq
    # print mem/1e9
    return rtn


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    # setup variables of interest
    q, n, qmax_bin, scatter_array, _, _ = setup_gpu_calc(atoms, sum_type)
    k_max = int((n ** 2 - n) / 2.)
    # break up problem
    pool_size = cpu_count()
    # pool_size = 4
    if pool_size <= 0:
        pool_size = 1
    p = Pool(pool_size, maxtasksperchild=1)
    tasks = []
    k_cov = 0
    while k_cov < k_max:
        m = atoms_pdf_gpu_fq(n, qmax_bin, 64e9 / 8.)
        if m > k_max - k_cov:
            m = k_max - k_cov
        task = (q, scatter_array, qbin, m, k_cov)
        tasks.append(task)
        k_cov += m
    # multiprocessing map problem
    fqs = p.map(atomic_fq, tasks)
    # p.join()
    p.close()
    # sum the answers
    final = np.sum(fqs, axis=0)
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    del q, n, qmax_bin, scatter_array, k_max, p, task, fqs
    return 2 * final


def atomic_grad_fq(task):
    q, scatter_array, qbin, k_max, k_cov = task
    n = len(q)

    qmax_bin = scatter_array.shape[1]

    d = np.empty((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)
    del q

    r = np.empty(k_max, np.float32)
    get_r_array(r, d)

    norm = np.empty((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)
    del scatter_array

    fq = np.empty((k_max, qmax_bin), np.float32)
    get_fq(fq, r, norm, qbin)

    grad = np.empty((k_max, 3, qmax_bin), np.float32)
    get_grad_fq(grad, fq, r, d, norm, qbin)
    del fq, r, d, norm

    # rtn = np.empty((n, 3, qmax_bin), np.float32)
    rtn = np.zeros((n, 3, qmax_bin), np.float32)
    fast_fast_flat_sum(rtn, grad, k_cov)
    del grad
    return rtn


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    # setup variables of interest
    q, n, qmax_bin, scatter_array, _, _ = setup_gpu_calc(atoms, sum_type)
    k_max = int((n ** 2 - n) / 2.)
    # break up problem
    pool_size = cpu_count()
    # pool_size = 4
    if pool_size <= 0:
        pool_size = 1
    p = Pool(pool_size, maxtasksperchild=1)
    tasks = []
    k_cov = 0
    while k_cov < k_max:
        m = atoms_per_gpu_grad_fq(n, qmax_bin, float(
            psutil.virtual_memory().available) / pool_size)
        if m > k_max - k_cov:
            m = k_max - k_cov
        task = (q, scatter_array, qbin, m, k_cov)
        tasks.append(task)
        k_cov += m
    # multiprocessing map problem
    fqs = p.map(atomic_grad_fq, tasks)
    # p.join()
    p.close()
    # sum the answers
    grad_p = np.sum(fqs, axis=0)
    # '''
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :] = np.nan_to_num(1 / na * grad_p[tx, tz, :])
    np.seterr(**old_settings)
    # '''
    del q, n, qmax_bin, scatter_array, k_max, p, task, fqs
    return grad_p


if __name__ == '__main__':
    from ase.atoms import Atoms
    from pyiid.wrappers.elasticscatter import wrap_atoms
    from pyiid.wrappers.cpu_wrappers.cpu_wrap import wrap_fq_grad as mfqg
    import matplotlib.pyplot as plt
    from numpy.testing import assert_allclose

    plt.ion()
    # n = 1000
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    # fq = wrap_fq(atoms, atoms.info['exp']['qbin'])
    # fq2 = mfq(atoms, atoms.info['exp']['qbin'])
    # print fq2.shape
    # plt.plot(fq)
    # plt.plot(fq2)
    # plt.plot((fq-fq2)/fq)
    # plt.show()
    # assert_allclose(fq, fq2, 3e-4)
    grad_fq = wrap_fq_grad(atoms, atoms.info['exp']['qbin'])
    mgrad_fq = mfqg(atoms, atoms.info['exp']['qbin'])
    assert_allclose(grad_fq, mgrad_fq)
    # raw_input()
