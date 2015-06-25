__author__ = 'christopher'
from numpy.testing import assert_allclose
from copy import deepcopy as dc
from numba import cuda

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
    q = np.random.random((n, 3)).astype(np.float32)
    cd = np.zeros((n, n, 3), dtype=np.float32)
    kd = dc(cd)

    # compiled version
    serial_get_d_array(cd, q)

    # kernel version
    il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    jl = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    get_ij_lists(il, jl, n)
    # # print len(il), len(jl)
    stream, bpg, tpb = set_up_gpu(len(il))
    qi = q[il]
    qj = q[jl]

    gpud = np.zeros((len(il), 3), dtype=np.float32)
    dgpud = cuda.to_device(gpud)
    dqi = cuda.to_device(qi)
    dqj = cuda.to_device(qj)

    get_d_array[bpg, tpb, stream](dgpud, dqi, dqj)
    dgpud.to_host()

    antisymmetric_reshape(kd, gpud, il, jl)
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

    il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    jl = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    get_ij_lists(il, jl, n)

    stream, bpg, tpb = set_up_gpu(len(il))
    qi = q[il]
    qj = q[jl]

    gpud = np.zeros((len(il), 3), dtype=np.float32)
    dgpud = cuda.to_device(gpud)
    dqi = cuda.to_device(qi)
    dqj = cuda.to_device(qj)

    get_d_array[bpg, tpb, stream](dgpud, dqi, dqj)
    dgpud.to_host()

    antisymmetric_reshape(kd, gpud, il, jl)
    assert_allclose(cd, kd)

    r = np.zeros(len(il), dtype=np.float32)
    dr = cuda.to_device(r)

    get_r_array[bpg, tpb, stream](dr, dgpud)
    dr.to_host()

    sr = np.zeros((n, n))
    cpur = np.zeros((n, n))
    symmetric_reshape(sr, r, il, jl)

    serial_get_r_array(cpur, cd)
    assert_allclose(sr, cpur)
    return

if __name__ == '__main__':
    import nose

    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
