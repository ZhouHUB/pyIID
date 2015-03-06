__author__ = 'christopher'

import numpy as np
from numpy.testing import assert_allclose

from pyiid.kernels.serial_kernel import *


def test_get_d_array():
    """
    Test of get_d_array
    """

    # prep data
    n = 60
    q = np.random.random((n, 3)).astype(np.float32)

    kd = np.zeros((n, n, 3))
    nd = np.zeros((n, n, 3))

    # numpy version
    for tx in range(n):
        for ty in range(n):
            for tz in range(3):
                nd[tx, ty, tz] = q[ty, tz] - q[tx, tz]

    # kernel version
    get_d_array(kd, q)
    assert_allclose(nd, kd)

    return


def test_get_r_array():
    """
    Test of get_r_array
    """

    # prep data
    n = 60
    d = np.random.random((n, n, 3)).astype(np.float32)

    kr = np.zeros((n, n), dtype=np.float32)

    # numpy version
    a = d**2
    b = a.sum(axis=2)
    nr = np.sqrt(b)

    # kernel version
    get_r_array(kr, d)
    # print nr-kr
    assert(type(kr) == type(nr))
    assert_allclose(nr, kr, rtol=1e-6)

    return


def test_get_scatter_array():

    scatter_array = np.loadtxt('pyIId/pyiid/tests/test_kernels/c60_scat.txt', dtype=np.float32)
    ksa = np.zeros(scatter_array.shape)
    numbers = np.ones(len(ksa), dtype=np.int)*6
    qbin = .1
    get_scatter_array(ksa, numbers, qbin)

    assert_allclose(scatter_array, ksa, rtol=1e-2)


def test_fq_array():

    n = 60
    scatter_array = np.random.random((n, 250))
    r = np.random.random((n, n))
    qbin = .1

    fq = np.zeros(250)
    qmax_bin = len(fq)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    fq[kq] += scatter_array[tx, kq] * \
                              scatter_array[ty, kq] / \
                              r[tx, ty] * \
                              math.sin(kq * qbin * r[tx, ty])
    kfq = np.zeros(fq.shape)
    get_fq_array(kfq, r, scatter_array, qbin)
    assert_allclose(kfq, fq)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)