__author__ = 'christopher'
import numpy as np
from numpy.testing import assert_allclose
from copy import deepcopy as dc
import sys
sys.path.extend(['/mnt/work-data/dev/pyIID'])

from pyiid.kernels.multi_cuda import *
n = 1000


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
            assert(bpg*tpb >= e_dim)
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
            assert(bpg*tpb >= e_dim)
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
    get_d_array1[bpg, tpb, stream](dd, dq, 0)
    get_d_array2[bpg, tpb, stream](dd, 0)
    dd.to_host(stream)

    assert_allclose(cd, kd)

    return


def test_get_r_array():
    """
    Test of get_d_array
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
    get_r_array1[bpg, tpb, stream](d_out, d_in)
    get_r_array2[bpg, tpb, stream](d_out)
    d_out.to_host(stream)

    assert_allclose(cr, k_r)

    return


def test_get_normalization_array():
    from pyiid.kernels.cpu_kernel import get_normalization_array as comp
    Q = 250
    scat = np.random.random((n, Q)).astype(np.float32)
    c_norm = np.zeros((n, n, Q), dtype=np.float32)
    k_norm = dc(c_norm)

    #compiled
    comp(c_norm, scat)

    #kernel
    stream, bpg, tpb = set_up_gpu(n, 250)
    dnorm = cuda.to_device(k_norm)
    dscat = cuda.to_device(scat)
    get_normalization_array1[bpg, tpb, stream](dnorm, dscat, 0)
    get_normalization_array2[bpg, tpb, stream](dnorm, 0)
    dnorm.to_host(stream)

    assert_allclose(c_norm, k_norm)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)