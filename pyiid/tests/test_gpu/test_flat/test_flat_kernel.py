__author__ = 'christopher'
from numpy.testing import assert_allclose
from copy import deepcopy as dc
from numba import cuda
import numpy as np

cuda.select_device(0)
from pyiid.kernels.flat_kernel import *


def set_up_gpu(n, qmax_bin=None):
    stream = cuda.stream()
    # two kinds of test_kernels; NxN or NxNxQ
    if qmax_bin is None:
        # NXN
        elements_per_dim_1 = [n]
        tpb_l_1 = [32]
        bpg_l_1 = []
        for e_dim, tpb in zip(elements_per_dim_1, tpb_l_1):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_1.append(bpg)
        return stream, bpg_l_1, tpb_l_1

    else:
        # NxNxQ
        elements_per_dim_2 = [n, qmax_bin]
        tpb_l_2 = [16, 2]
        bpg_l_2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_2.append(bpg)
        return stream, bpg_l_2, tpb_l_2


def test_get_d():
    from pyiid.kernels.cpu_kernel import get_d_array as serial_get_d_array
    # prep data
    n = 4
    k = n * (n - 1) / 2
    q = np.random.random((n, 3)).astype(np.float32)
    cd = np.zeros((n, n, 3), dtype=np.float32)
    kd = dc(cd)

    # compiled version
    serial_get_d_array(cd, q)

    # kernel version
    stream, bpg, tpb = set_up_gpu(k)
    gpud = np.zeros((k, 3), dtype=np.float32)
    dgpud = cuda.to_device(gpud)
    dq = cuda.to_device(q)

    get_d_array[bpg, tpb, stream](dgpud, dq, 0)
    dgpud.to_host()

    antisymmetric_reshape(kd, gpud)
    print cd.shape, kd.shape
    assert_allclose(cd, kd)

    return


def test_get_r():
    from pyiid.kernels.cpu_kernel import get_d_array as serial_get_d_array
    from pyiid.kernels.cpu_kernel import get_r_array as serial_get_r_array
    # prep data
    n = 4
    q = np.random.random((n, 3)).astype(np.float32)
    cd = np.zeros((n, n, 3), dtype=np.float32)
    kd = dc(cd)

    # compiled version
    serial_get_d_array(cd, q)

    # kernel version
    k = n * (n - 1) / 2.
    stream, bpg, tpb = set_up_gpu(k)

    gpud = np.zeros((k, 3), dtype=np.float32)
    dgpud = cuda.to_device(gpud)
    dq = cuda.to_device(q)

    get_d_array[bpg, tpb, stream](dgpud, dq, 0)
    dgpud.to_host()

    antisymmetric_reshape(kd, gpud)
    assert_allclose(cd, kd)

    r = np.zeros(k, dtype=np.float32)
    dr = cuda.to_device(r)

    get_r_array[bpg, tpb, stream](dr, dgpud)
    dr.to_host()

    sr = np.zeros((n, n))
    cpur = np.zeros((n, n))
    symmetric_reshape(sr, r)

    serial_get_r_array(cpur, cd)
    assert_allclose(sr, cpur)
    return


if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
