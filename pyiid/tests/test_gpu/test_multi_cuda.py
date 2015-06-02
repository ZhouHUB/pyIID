__author__ = 'christopher'
import numpy as np
from numpy.testing import assert_allclose
from copy import deepcopy as dc
import sys

sys.path.extend(['/mnt/work-data/dev/pyIID'])

from pyiid.kernels.multi_cuda import *

n = 60


def set_up_gpu(n, qmax_bin=None):
    stream = cuda.stream()
    # two kinds of test_kernels; NxN or NxNxQ
    if qmax_bin is None:
        # NXN
        elements_per_dim_2 = [n, n]
        tpb_l_2 = [32, 32]
        bpg_l_2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_2.append(bpg)
        return stream, bpg_l_2, tpb_l_2

    else:
        # NxNxQ
        elements_per_dim_3 = [n, n, qmax_bin]
        tpb_l_3 = [16, 16, 4]
        bpg_l_3 = []
        for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_3.append(bpg)
        return stream, bpg_l_3, tpb_l_3


def test_get_d_array():
    """
    Test of cuda get_d_array
    """
    from pyiid.kernels.cpu_kernel import get_d_array as serial_get_d_array
    # prep data
    # n = 600
    q = np.random.random((n, 3)).astype(np.float32)
    cd = np.zeros((n, n, 3), dtype=np.float32)
    kd = dc(cd)

    # compiled version
    serial_get_d_array(cd, q)

    # kernel version
    stream, bpg, tpb = set_up_gpu(n)

    dd = cuda.to_device(kd, stream)
    dq = cuda.to_device(q, stream)
    get_d_array[bpg, tpb, stream](dd, dq, 0)
    dd.to_host(stream)

    assert_allclose(cd, kd)

    return


def test_get_r_array():
    """
    Test of get_r_array
    """
    from pyiid.kernels.cpu_kernel import get_r_array as comp
    from pyiid.kernels.cpu_kernel import get_d_array as serial_get_d_array
    # prep data
    # n = 600
    q = np.random.random((n, 3)).astype(np.float32)
    cd = np.zeros((n, n, 3), dtype=np.float32)

    # compiled version
    serial_get_d_array(cd, q)
    # d = np.random.random((n, n, 3)).astype(np.float32)
    d = cd
    cr = np.zeros((n, n), dtype=np.float32)
    k_r = dc(cr)

    # compiled version
    comp(cr, d)

    # kernel version
    stream, bpg, tpb = set_up_gpu(n)
    d_out = cuda.to_device(k_r, stream)
    d_in = cuda.to_device(d, stream)
    get_r_array[bpg, tpb, stream](d_out, d_in)
    d_out.to_host(stream)

    assert_allclose(cr, k_r)

    return


def test_get_normalization_array():
    from pyiid.kernels.cpu_kernel import get_normalization_array as comp

    Q = 250
    scat = np.random.random((n, Q)).astype(np.float32)
    c_norm = np.zeros((n, n, Q), dtype=np.float32)
    k_norm = dc(c_norm)

    # compiled
    comp(c_norm, scat)

    #kernel
    stream, bpg, tpb = set_up_gpu(n, 250)
    dnorm = cuda.to_device(k_norm)
    dscat = cuda.to_device(scat)
    get_normalization_array[bpg, tpb, stream](dnorm, dscat, 0)
    dnorm.to_host(stream)

    assert_allclose(c_norm, k_norm)


def test_get_fq_p0_1():
    Q = 250
    r = np.random.random((n, n)).astype(np.float32)
    qbin = .1
    cfq = np.zeros((n, n, Q), dtype=np.float32)
    # compiled
    qmax_bin = cfq.shape[-1]
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    cfq[tx, ty, kq] = math.sin(kq * qbin * r[tx, ty]) / r[
                        tx, ty]

    #kernel
    kfq = np.zeros((n, n, Q), dtype=np.float32)
    dfq = cuda.to_device(kfq)
    dr = cuda.to_device(r)
    stream, bpg, tpb = set_up_gpu(n, 250)
    get_fq_step_0[bpg, tpb, stream](dfq, dr, qbin)
    dfq.to_host(stream)

    rtol = 5e-2
    print 'rms', np.sqrt(np.mean((kfq - cfq) ** 2))
    print 'mean', np.mean(kfq - cfq)
    print 'median', np.median(kfq - cfq)
    print 'percent of errors', np.count_nonzero(
        kfq - cfq > cfq * rtol) / float(kfq.size) * 100, '%'
    assert_allclose(kfq, cfq, rtol=rtol)


def test_get_fq_p0_1_sum():
    Q = 250
    r = np.random.random((n, n)).astype(np.float32)
    # r = np.random.random((n, n)).astype(np.float64)
    qbin = .1
    cfq = np.zeros((n, n, Q), dtype=np.float32)
    # cfq = np.zeros((n, n, Q), dtype=np.float64)
    # compiled
    qmax_bin = cfq.shape[-1]
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    cfq[tx, ty, kq] = math.sin(kq * qbin * r[tx, ty]) / r[
                        tx, ty]

    #kernel
    # kfq = np.zeros((n, n, Q), dtype=np.float64)
    kfq = np.zeros((n, n, Q), dtype=np.float32)
    dfq = cuda.to_device(kfq)
    dr = cuda.to_device(r)
    stream, bpg, tpb = set_up_gpu(n, 250)
    get_fq_step_0[bpg, tpb, stream](dfq, dr, qbin)
    dfq.to_host(stream)

    rtol = 5e-2
    print 'rms', np.sqrt(np.mean((kfq - cfq) ** 2))
    print 'mean', np.mean(kfq - cfq)
    print 'median', np.median(kfq - cfq)
    print 'percent of errors', np.count_nonzero(
        kfq - cfq > cfq * rtol) / float(kfq.size) * 100, '%'
    assert_allclose(kfq.sum(axis=(0, 1)), cfq.sum(axis=(0, 1)), rtol=rtol)


def test_get_fq_grad_p3():
    Q = 250
    r = np.random.random((n, n)).astype(np.float32)
    # r = np.random.random((n, n)).astype(np.float64)
    qbin = .1
    cfq = np.zeros((n, n, Q), dtype=np.float32)
    # cfq = np.zeros((n, n, Q), dtype=np.float64)
    # compiled
    qmax_bin = cfq.shape[-1]
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    cfq[tx, ty, kq] = math.cos(
                        kq * qbin * r[tx, ty]) * kq * qbin / r[
                                          tx, ty]

    #kernel
    # kfq = np.zeros((n, n, Q), dtype=np.float64)
    kfq = np.zeros((n, n, Q), dtype=np.float32)
    dfq = cuda.to_device(kfq)
    dr = cuda.to_device(r)
    stream, bpg, tpb = set_up_gpu(n, 250)
    fq_grad_step_0[bpg, tpb, stream](dfq, dr, qbin)
    dfq.to_host(stream)

    rtol = 5e-2
    print 'rms', np.sqrt(np.mean((kfq - cfq) ** 2))
    print 'mean', np.mean(kfq - cfq)
    print 'median', np.median(kfq - cfq)
    print 'percent of errors', np.count_nonzero(
        kfq - cfq > cfq * rtol) / float(kfq.size) * 100, '%'
    assert_allclose(kfq, cfq, rtol=rtol)


def test_get_fq_grad_p3_sum():
    Q = 250
    r = np.random.random((n, n)).astype(np.float32)
    # r = np.random.random((n, n)).astype(np.float64)
    qbin = .1
    cfq = np.zeros((n, n, Q), dtype=np.float32)
    # cfq = np.zeros((n, n, Q), dtype=np.float64)
    # compiled
    qmax_bin = cfq.shape[-1]
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    cfq[tx, ty, kq] = math.cos(
                        kq * qbin * r[tx, ty]) * kq * qbin / r[
                                          tx, ty]

    #kernel
    # kfq = np.zeros((n, n, Q), dtype=np.float64)
    kfq = np.zeros((n, n, Q), dtype=np.float32)
    dfq = cuda.to_device(kfq)
    dr = cuda.to_device(r)
    stream, bpg, tpb = set_up_gpu(n, 250)
    fq_grad_step_0[bpg, tpb, stream](dfq, dr, qbin)
    dfq.to_host(stream)

    rtol = 5e-2
    print 'rms', np.sqrt(np.mean((kfq - cfq) ** 2))
    print 'mean', np.mean(kfq - cfq)
    print 'median', np.median(kfq - cfq)
    print 'percent of errors', np.count_nonzero(
        kfq - cfq > cfq * rtol) / float(kfq.size) * 100, '%'
    assert_allclose(kfq.sum(axis=(0, 1)), cfq.sum(axis=(0, 1)), rtol=rtol)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
    '''
    Q = 250
    n = 200
    if n == 2:
        r = np.asarray([[0, 1], [1, 0]]).astype(np.float32)
    else:
        r = np.random.random((n, n)).astype(np.float32) * 10
    qbin = .1
    cfq = np.zeros((n, n, Q), dtype=np.float32)
    # CPU compiled
    qmax_bin = cfq.shape[-1]
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    cfq[tx, ty, kq] = math.sin(kq * qbin * r[tx, ty]) / r[
                        tx, ty]

    # GPU kernel
    kfq = np.zeros((n, n, Q), dtype=np.float32)
    dfq = cuda.to_device(kfq)
    dr = cuda.to_device(r)
    stream, bpg, tpb = set_up_gpu(n, 250)
    get_fq_step_0[bpg, tpb, stream](dfq, dr, qbin)
    dfq.to_host(stream)
    cuda.close()

    import matplotlib.pyplot as plt

    plt.plot(kfq.sum(axis=(0, 1)), label='gpu')
    plt.plot(cfq.sum(axis=(0, 1)), label='cpu')
    plt.legend()
    plt.show()'''