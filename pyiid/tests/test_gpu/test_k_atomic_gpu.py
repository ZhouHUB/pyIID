__author__ = 'christopher'
from pyiid.wrappers import generate_grid
import math
from numba import cuda
import numpy as np
from pyiid.wrappers.k_atomic_gpu import atoms_pdf_gpu_fq, atoms_per_gpu_grad_fq
from pyiid.wrappers.elasticscatter import wrap_atoms
from ase.atoms import Atoms
from pyiid.kernels.flat_kernel import get_ij_lists
from numpy.testing import assert_allclose


def generate_qij(qi, qj, q, il, jl):
    for k in xrange(len(il)):
        for tz in range(3):
            qi[k, tz] = q[il[k], tz]
            qj[k, tz] = q[jl[k], tz]

def generate_d(d, qi, qj):
    for k in xrange(len(d)):
        for tz in range(3):
            d[k, tz] = qi[k, tz] - qj[k, tz]

def generate_r(r, d):
    for k in xrange(len(r)):
        r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)

def generate_scatij(scati, scatj, scat, il, jl):
    for k in xrange(len(il)):
        for qx in xrange(scat.shape[1]):
            scati[k, qx] = scat[il[k], qx]
            scatj[k, qx] = scat[jl[k], qx]

def generate_norm(norm, scati, scatj):
    for k in xrange(norm.shape[0]):
        for qx in xrange(norm.shape[1]):
            norm[k, qx] = scati[k, qx] * scatj[k, qx]

def generate_fq(fq, r, norm, qbin):
    for k in xrange(fq.shape[0]):
        for qx in xrange(fq.shape[1]):
            fq[k, qx] = norm[k, qx] * math.sin(qbin * qx * r[k]) / r[k]

def test_fq():
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    n = len(atoms)
    wrap_atoms(atoms)
    q = atoms.get_positions().astype(np.float32)
    scatter_array = atoms.get_array('F(Q) scatter')
    qbin = atoms.info['exp']['qbin']

    il = np.zeros((n ** 2 - n) / 2., dtype=np.uint32)
    jl = il.copy()
    get_ij_lists(il, jl, n)

    k_max = len(il)
    qmax_bin = scatter_array.shape[1]
        # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, d2_to_d1_sum, construct_qij, \
        construct_scatij

    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [32]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_q = [qmax_bin]
    tpbq = [4]
    bpgq = generate_grid(elements_per_dim_q, tpbq)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [16, 4]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    # generate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dqi = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dqj = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dil = cuda.to_device(np.asarray(il, dtype=np.uint32), stream=stream)
    djl = cuda.to_device(np.asarray(jl, dtype=np.uint32), stream=stream)
    dq = cuda.to_device(q)

    # calculate kernels
    # test qij
    construct_qij[bpg1, tpb1, stream](dqi, dqj, dq, dil, djl)
    test1 = np.empty(dqi.shape, np.float32)
    test2 = np.empty(dqj.shape, np.float32)
    dqi.copy_to_host(test1)
    dqj.copy_to_host(test2)

    qi = np.empty(test1.shape)
    qj = np.empty(test2.shape)
    generate_qij(qi, qj, q, il, jl)

    assert_allclose(test1, qi)
    assert_allclose(test2, qj)

    # test dd
    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)
    get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
    del dqi, dqj

    gpu_d = np.empty(dd.shape, np.float32)
    dd.copy_to_host(gpu_d)
    d = np.empty(dd.shape)
    generate_d(d, qi, qj)
    assert_allclose(gpu_d, d)

    # test dr
    get_r_array[bpg1, tpb1, stream](dr, dd)

    gpu_r = np.empty(dr.shape, np.float32)
    dr.copy_to_host(gpu_r)
    r = np.empty(dr.shape)
    generate_r(r, d)
    assert_allclose(gpu_r, r)


    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscati = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscatj = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)

    # test scatij
    construct_scatij[bpg2, tpb2, stream2](dscati, dscatj, dscat, dil, djl)

    gpu_scati = np.empty(dscati.shape, np.float32)
    gpu_scatj = np.empty(dscatj.shape, np.float32)
    dscati.copy_to_host(gpu_scati)
    dscatj.copy_to_host(gpu_scatj)
    scati = np.empty(dscati.shape, np.float32)
    scatj = np.empty(dscatj.shape, np.float32)
    generate_scatij(scati, scatj, scatter_array, il, jl)
    assert_allclose(gpu_scati, scati)
    assert_allclose(gpu_scatj, scatj)

    # test dnorm
    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)

    gpu_norm = np.empty(dnorm.shape, np.float32)
    dnorm.copy_to_host(gpu_norm)
    norm = np.empty(dnorm.shape, np.float32)
    generate_norm(norm, scati, scatj)
    assert_allclose(gpu_norm, norm)

    del dscati, dscatj, dscat, dil, djl
    # test F(Q)
    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    final = np.zeros(qmax_bin, dtype=np.float32)
    dfinal = cuda.to_device(final)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    gpu_fq = np.empty(dfq.shape, np.float32)
    dfq.copy_to_host(gpu_fq)
    fq = np.empty(dfq.shape, np.float32)
    generate_fq(fq, r, norm, qbin)
    assert_allclose(fq, gpu_fq, atol=5e-4)

    del dr, dnorm
    d2_to_d1_sum[bpgq, tpbq, stream2](dfinal, dfq)
    del dfq

    dfinal.to_host(stream2)
    del dfinal
    return final


def test_grad_fq(q, scatter_array, qbin, il, jl):
    k_max = len(il)
    qmax_bin = scatter_array.shape[1]

    # load kernels
    from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
        get_normalization_array, get_fq, get_grad_fq, \
        construct_scatij, construct_qij, flat_sum, zero_pseudo_3D

    # generate grids
    elements_per_dim_1 = [k_max]
    tpb1 = [64]
    bpg1 = generate_grid(elements_per_dim_1, tpb1)

    elements_per_dim_q = [qmax_bin]
    tpbq = [4]
    bpgq = generate_grid(elements_per_dim_q, tpbq)

    elements_per_dim_2 = [k_max, qmax_bin]
    tpb2 = [16, 4]
    bpg2 = generate_grid(elements_per_dim_2, tpb2)

    elements_per_dim_nq = [len(q), qmax_bin]
    tpbnq = [16, 4]
    bpgnq = generate_grid(elements_per_dim_nq, tpbnq)

    # gnerate streams
    stream = cuda.stream()
    stream2 = cuda.stream()

    # transfer data
    dd = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dq = cuda.to_device(q, stream=stream)
    dqi = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dqj = cuda.device_array((k_max, 3), dtype=np.float32, stream=stream)
    dil = cuda.to_device(np.asarray(il, dtype=np.uint32), stream=stream)
    djl = cuda.to_device(np.asarray(jl, dtype=np.uint32), stream=stream)

    construct_qij[bpg1, tpb1, stream](dqi, dqj, dq, dil, djl)
    dr = cuda.device_array(k_max, dtype=np.float32, stream=stream)

    # calculate kernels
    get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
    del dqi, dqj

    get_r_array[bpg1, tpb1, stream](dr, dd)

    dnorm = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscati = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscatj = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)
    dscat = cuda.to_device(scatter_array, stream=stream2)

    construct_scatij[bpg2, tpb2, stream2](dscati, dscatj, dscat, dil, djl)
    get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)
    del dscati, dscatj, dscat

    dfq = cuda.device_array((k_max, qmax_bin), dtype=np.float32, stream=stream2)

    get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

    grad = np.zeros((k_max, 3, qmax_bin), dtype=np.float32)
    dgrad = cuda.device_array(grad.shape, dtype=np.float32, stream=stream2)

    get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)

    new_grad2 = np.zeros((len(q), 3, qmax_bin), dtype=np.float32)

    dnew_grad = cuda.device_array(new_grad2.shape, dtype=np.float32,
                                  stream=stream2)
    zero_pseudo_3D[bpgnq, tpbnq, stream2](dnew_grad)

    flat_sum[[1, bpgq[0]], [4, tpbq[0]], stream2](dnew_grad, dgrad, dil,
                                                  djl)
    dnew_grad.copy_to_host(new_grad2)
    del dd, dr, dnorm, dfq, dgrad, dil, djl
    del grad, il, jl
    return new_grad2

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
