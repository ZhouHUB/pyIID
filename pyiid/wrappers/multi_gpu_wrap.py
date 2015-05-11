__author__ = 'christopher'
import numpy as np
from numba import cuda
import math
from threading import Thread
from numpy.testing import assert_allclose


def sub_fq(gpu, q, scatter_array, fq_q, qmax_bin, qbin, m, n_cov):
    print 'start sub'
    n = len(q)
    tups = [(m, n, 3), (m, n), (m, n, qmax_bin), (m, n, qmax_bin)]
    data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
    # Kernel
    # cuda kernel information
    # with gpu:
    cuda.select_device(gpu)
    from pyiid.kernels.multi_cuda import get_d_array1, get_d_array2, \
        get_normalization_array1,get_normalization_array2, get_r_array1, \
        get_r_array2, get_fq_p0, \
        get_fq_p1, get_fq_p3, gpu_reduce_3D_to_1D, gpu_reduce_3D_to_2D, \
        gpu_reduce_2D_to_1D

    stream = cuda.stream()
    stream2 = cuda.stream()

    print 'define grids'
    # three kinds of test_kernels; Q, NxN or NxNxQ
    # Q
    elements_per_dim_1 = [qmax_bin]
    tpb_l_1 = [32]
    bpg_l_1 = []
    for e_dim, tpb in zip(elements_per_dim_1, tpb_l_1):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_1.append(bpg)
    # NxQ
    elements_per_dim_2_q = [n, qmax_bin]
    tpb_l_2_q = [32, 32]
    bpg_l_2_q = []
    for e_dim, tpb in zip(elements_per_dim_2_q, tpb_l_2_q):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2_q.append(bpg)

    # NXN
    elements_per_dim_2 = [m, n]
    tpb_l_2 = [32, 32]
    # tpb_l_2 = [8, 4]
    bpg_l_2 = []
    for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_2.append(bpg)

    # NxNxQ
    elements_per_dim_3 = [m, n, qmax_bin]
    tpb_l_3 = [16, 16, 4]
    # tpb_l_3 = [4, 4, 2]
    bpg_l_3 = []
    for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
        bpg = int(math.ceil(float(e_dim) / tpb))
        if bpg < 1:
            bpg = 1
        assert (bpg * tpb >= e_dim)
        bpg_l_3.append(bpg)

    # start calculations
    print 'start calculation'
    dscat = cuda.to_device(scatter_array, stream2)
    dnorm = cuda.device_array(data[2].shape, dtype=np.float32,
                              stream=stream2)
    dscat.to_host()
    dnorm.copy_to_host(data[2])
    print n_cov
    print dnorm.shape
    print dscat.shape
    print 'gpu error after this'
    # get_normalization_array1[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)
    get_normalization_array1[bpg_l_3, tpb_l_3, stream2](dnorm, dscat)

    # get_normalization_array2[bpg_l_3, tpb_l_3, stream2](dnorm, n_cov)



    dd = cuda.device_array(data[0].shape, dtype=np.float32, stream=stream)
    dr = cuda.device_array(data[1].shape, dtype=np.float32, stream=stream)
    # dfq = cuda.device_array(data[3].shape, dtype=np.float32, stream=stream)
    assert not np.all(data[3])
    dfq = cuda.to_device(data[3], stream=stream)
    dq = cuda.to_device(q, stream)

    print 'error continues'
    get_d_array1[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
    # get_d_array2[bpg_l_2, tpb_l_2, stream](dd, n_cov)
    print 'got d array'
    get_r_array1[bpg_l_2, tpb_l_2, stream](dr, dd)
    # get_r_array2[bpg_l_2, tpb_l_2, stream](dr)

    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final, stream2)

    # dfinal_2D = cuda.device_array((n, qmax_bin), dtype=np.float32, stream=stream2)

    get_fq_p0[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
    get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
    get_fq_p1[bpg_l_3, tpb_l_3, stream](dfq)


    # gpu_reduce_3D_to_2D[bpg_l_2_q, tpb_l_2_q, stream](dfinal_2D, dfq)
    # gpu_reduce_2D_to_1D[bpg_l_1, tpb_l_1, stream](dfinal, dfinal_2D)

    gpu_reduce_3D_to_1D[bpg_l_1, tpb_l_1, stream](dfinal, dfq)


    dfinal.to_host(stream)
    # print final
    # dfq.to_host(stream)
    # final = data[3].sum(axis=(0, 1))
    fq_q.append(final)

    del data, dscat, dnorm, dd, dq, dr, dfq, final, dfinal
    cuda.close()
    return


def wrap_fq(atoms, qmax=25., qbin=.1):
    # print 'start wrapper'
    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(math.ceil(qmax / qbin))
    scatter_array = atoms.get_array('scatter')

    # get info on our gpu setup and memory requrements
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
    gpus = range(len(gpus))
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    gpu_total_mem = sum(sort_gmem)
    total_req_mem = (2 * qmax_bin * n * n + qmax_bin * n
                     + 4 * n * n + 3 * n + qmax_bin) * 4

    # starting buffers
    fq_q = []
    n_cov = 0
    p_dict = {}

    # TODO: NEEDS WORK GIVES BAD RESULTS WITH EQUAL WEIGHTING
    # if total_req_mem < gpu_total_mem:
    if False:
        # Then the total gpu space is larger than our problem, we should give
        # out atoms by the size of the free memory, giving each gpu some work
        # to do based on its capacity.
        gpu_m = [int(round(n * float(g_mem) / gpu_total_mem)) for g_mem in
                 sort_gmem]
        for gpu, m in zip(sort_gpus, gpu_m):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if n_cov >= n:
                    break
                p = Thread(
                    target=sub_fq, args=(
                        gpu, q, scatter_array,
                        fq_q, qmax_bin, qbin, m, n_cov))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov >= n:
                    break
        assert n_cov == n
    else:
        print 'start threading'
        # The total amount of work is greater than the sum of our GPUs, no
        # special distribution needed, just keep putting problems on GPUs until
        # finished.
        while n_cov < n:
            for gpu, mem in zip(sort_gpus, sort_gmem):
                m = int(
                    math.floor(
                        float(
                            -4 * n * qmax_bin - 12 * n - 4 * qmax_bin + .8 * mem) / (
                            8 * n * (qmax_bin + 2))))
                if m > n - n_cov:
                    m = n - n_cov

                if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                    if n_cov >= n:
                        break
                    p = Thread(
                        target=sub_fq, args=(
                            gpu, q, scatter_array,
                            fq_q, qmax_bin, qbin, m, n_cov))
                    p.start()
                    p_dict[gpu] = p
                    n_cov += m

                    if n_cov >= n:
                        break

    print 'start join'
    for value in p_dict.values():
        print value
        value.join()
    print 'finish threading'
    fq = np.zeros(qmax_bin)
    for ele in fq_q:
        fq[:] += ele
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def sub_grad(gpu, q, scatter_array, grad_q, qmax_bin, qbin, m, n_cov,
             index_list):
    # Build Data arrays
    n = len(q)
    tups = [(m, n, 3), (m, n),
            (m, n, qmax_bin), (m, n, qmax_bin),
            (m, n, 3, qmax_bin), (m, n, qmax_bin)]
    data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
    with gpu:
        from pyiid.kernels.multi_cuda import get_d_array1, get_d_array2, \
            get_normalization_array1, get_r_array1, get_r_array2, get_fq_p0, \
            get_fq_p1, get_fq_p3, gpu_reduce_4D_to_3D
        from pyiid.kernels.multi_cuda import fq_grad_position3, \
            fq_grad_position5, fq_grad_position7, fq_grad_position_final1, \
            fq_grad_position_final2
        # cuda info
        stream = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()

        # two kinds of test_kernels; NxN or NxNxQ
        # NXN
        elements_per_dim_2 = [m, n]
        tpb_l_2 = [32, 32]
        bpg_l_2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
            bpg_l_2.append(int(math.ceil(float(e_dim) / tpb)))

        # NxNxQ
        elements_per_dim_3 = [m, n, qmax_bin]
        tpb_l_3 = [16, 16, 4]
        bpg_l_3 = []
        for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
            bpg_l_3.append(int(math.ceil(float(e_dim) / tpb)))

        # START CALCULATIONS---------------------------------------------------
        dscat = cuda.to_device(scatter_array, stream2)
        dnorm = cuda.device_array(data[2].shape, dtype=np.float32,
                                  stream=stream2)
        '--------------------------------------------------------------'
        get_normalization_array1[bpg_l_3, tpb_l_3, stream2](dnorm, dscat,
                                                            n_cov)
        '--------------------------------------------------------------'
        dd = cuda.device_array(data[0].shape, dtype=np.float32, stream=stream)
        dr = cuda.device_array(data[1].shape, dtype=np.float32, stream=stream)
        dfq = cuda.device_array(data[3].shape, dtype=np.float32, stream=stream)
        dq = cuda.to_device(q, stream)

        get_d_array1[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
        get_d_array2[bpg_l_2, tpb_l_2, stream](dd, n_cov)

        get_r_array1[bpg_l_2, tpb_l_2, stream](dr, dd)
        get_r_array2[bpg_l_2, tpb_l_2, stream](dr)

        '--------------------------------------------------------------'
        get_fq_p0[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
        get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
        '--------------------------------------------------------------'
        dcos_term = cuda.device_array(data[5].shape, dtype=np.float32,
                                      stream=stream2)
        # cuda.synchronize()


        get_fq_p1[bpg_l_3, tpb_l_3, stream](dfq)
        fq_grad_position3[bpg_l_3, tpb_l_3, stream3](dcos_term, dr, qbin)
        dgrad_p = cuda.device_array(data[4].shape, dtype=np.float32,
                                    stream=stream2)
        # cuda.synchronize()


        # cuda.synchronize()
        final = np.zeros((m, 3, qmax_bin), dtype=np.float32)
        dfinal = cuda.to_device(final, stream=stream3)

        fq_grad_position_final1[bpg_l_3, tpb_l_3, stream](dgrad_p, dd, dr)
        fq_grad_position5[bpg_l_3, tpb_l_3, stream2](dcos_term, dnorm)
        # cuda.synchronize()

        fq_grad_position7[bpg_l_3, tpb_l_3, stream](dcos_term, dfq, dr)
        # cuda.synchronize()

        fq_grad_position_final2[bpg_l_3, tpb_l_3, stream](dgrad_p, dcos_term)

        gpu_reduce_4D_to_3D[bpg_l_3, tpb_l_3, stream](dfinal, dgrad_p)

        dfinal.to_host(stream)
        grad_q.append(final)
        index_list.append(n_cov)
        del data, dscat, dnorm, dd, dr, dfq, dcos_term, dgrad_p


def wrap_fq_grad(atoms, qmax=25., qbin=.1):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    gpu_total_mem = sum(sort_gmem)
    total_req_mem = (
                        6 * qmax_bin * n * n + qmax_bin * n + 4 * n * n + 3 * n + qmax_bin * 3 * n) * 4

    grad_q = []
    index_list = []
    p_dict = {}
    n_cov = 0


    # if total_req_mem < gpu_total_mem:
    if False:
        # Then the total gpu space is larger than our problem, we should give
        # out atoms by the size of the free memory, giving each gpu some work
        # to do based on its capacity.
        gpu_m = [int(round(n * float(g_mem) / gpu_total_mem)) for g_mem in
                 sort_gmem]
        for gpu, m in zip(sort_gpus, gpu_m):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                p = Thread(
                    target=sub_grad, args=(
                        gpu, q, scatter_array,
                        grad_q, qmax_bin, qbin, m, n_cov, index_list))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov == n:
                    break
        assert n_cov == n
    else:
        while n_cov < n:
            for gpu, mem in zip(sort_gpus, sort_gmem):
                # m = int(
                # math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                # 8 * n * (3 * qmax_bin + 2))))
                m = int(
                    math.floor(float(-4 * n * qmax_bin - 12 * n + .7 * mem) / (
                        4 * (6 * qmax_bin * n + 3 * qmax_bin + 4 * n))))
                if m > n - n_cov:
                    m = n - n_cov
                if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                    p = Thread(
                        target=sub_grad, args=(
                            gpu, q, scatter_array,
                            grad_q, qmax_bin, qbin, m, n_cov, index_list))
                    p_dict[gpu] = p
                    p.start()
                    n_cov += m
                    if n_cov == n:
                        break
    for value in p_dict.values():
        value.join()
    # Sort grads to make certain indices are in order
    sort_grads = [x for (y, x) in sorted(zip(index_list, grad_q))]

    len(sort_grads)
    # sorted_sum_grads = [x.sum(axis=(1)) for x in sort_grads]

    # Stitch arrays together
    if len(sort_grads) > 1:
        grad_p_final = np.concatenate(sort_grads, axis=0)
    else:
        grad_p_final = sort_grads[0]


    # sum down to 1D array
    grad_p = grad_p_final

    # sum reduce to 1D
    na = np.average(scatter_array, axis=0) ** 2 * n

    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            # grad_p[tx, tz, :qmin_bin] = 0.0
            grad_p[tx, tz] = np.nan_to_num(
                1 / na * grad_p[tx, tz])
    np.seterr(**old_settings)
    return grad_p


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.master_wrap import wrap_atoms
    import matplotlib.pyplot as plt

    n = 400
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au' + str(n), pos)
    # atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms, None)

    fq = wrap_fq(atoms)
    # grad_fq = wrap_fq_grad(atoms)
    # print grad_fq