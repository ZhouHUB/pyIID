__author__ = 'christopher'
import numpy as np
from multiprocessing import Pool, cpu_count
import math
from pyiid.kernels.multi_flat_cpu_kernel import *
from pyiid.wrappers.k_atomic_gpu import atoms_pdf_gpu_fq


def fq_mem(n, k, qmax_bin):
    q = n * 3
    d = k * 3
    r = k
    scat = n * qmax_bin
    norm = k * qmax_bin
    fq = k * qmax_bin
    rtn = qmax_bin
    return (q + d + r + scat + norm + fq + rtn) * 4


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
    mem = q.nbytes + scatter_array.nbytes

    d = np.empty((k_max, 3), np.float32)
    get_d_array(d, q, k_cov)
    mem += d.nbytes
    del q

    r = np.empty(k_max, np.float32)
    get_r_array(r, d)
    mem += r.nbytes
    del d

    norm = np.empty((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, k_cov)
    mem += norm.nbytes
    del scatter_array

    fq = np.empty((k_max, qmax_bin), np.float32)
    get_fq(fq, r, norm, qbin)
    mem += fq.nbytes
    del norm, r

    rtn = fq.sum(axis=0)
    mem += rtn.nbytes
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
        m = atoms_pdf_gpu_fq(n, qmax_bin, 50e9 / 8.)
        if m > k_max - k_cov:
            m = k_max - k_cov
        task = (q, scatter_array, qbin, m, k_cov)
        tasks.append(task)
        k_cov += m
    # multiprocessing map problem
    fqs = p.map(atomic_fq, tasks)
    # sum the answers
    final = np.sum(fqs, axis=0)
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    return 2 * final


if __name__ == '__main__':
    from ase.atoms import Atoms
    from pyiid.wrappers.elasticscatter import wrap_atoms
    from pyiid.wrappers.cpu_wrap import wrap_fq as mfq
    import matplotlib.pyplot as plt
    from numpy.testing import assert_allclose

    plt.ion()
    n = 10000
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au' + str(n), pos)
    # atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    fq = wrap_fq(atoms, atoms.info['exp']['qbin'])
    # fq2 = mfq(atoms, atoms.info['exp']['qbin'])
    # print fq2.shape
    plt.plot(fq)
    # plt.plot(fq2)
    # plt.plot((fq-fq2)/fq)
    plt.show()
    # assert_allclose(fq, fq2, 3e-4)
    # grad_fq = wrap_fq_grad(atoms, atoms.info['exp']['qbin'])
    # print grad_fq[:, :, 1]
    # raw_input()
