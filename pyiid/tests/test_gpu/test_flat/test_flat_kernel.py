__author__ = 'christopher'
from pyiid.tests import *
from copy import deepcopy as dc
from numba import cuda
import numpy as np
import matplotlib.pyplot as plt

gpus = cuda.gpus.lst
gpu = gpus[1]

from pyiid.kernels.flat_kernel import get_d_array as gd
from pyiid.kernels.multi_flat_cpu_kernel import get_d_array as cd

from pyiid.kernels.flat_kernel import get_r_array as gr
from pyiid.kernels.multi_flat_cpu_kernel import get_r_array as cr

from pyiid.kernels.flat_kernel import get_normalization_array as gn
from pyiid.kernels.multi_flat_cpu_kernel import get_normalization_array as cn

from pyiid.kernels.flat_kernel import get_fq as gfq
from pyiid.kernels.multi_flat_cpu_kernel import get_fq as cfq

from pyiid.wrappers import generate_grid
from pyiid.kernels import k_to_ij

n = 600
atoms = setup_atoms(n)

q = atoms.positions.astype(np.float32)
scatter_array = atoms.get_array('F(Q) scatter').astype(np.float32)
qmax_bin = scatter_array.shape[1]
k_max = n * (n - 1) / 2.
k_cov = 0
i_min = k_to_ij(k_cov)[0] - 1
i_max, _ = k_to_ij(k_cov + k_max)
offset = 0
elements_per_dim_1 = [k_max]
tpb1 = [64]
bpg1 = generate_grid(elements_per_dim_1, tpb1)

elements_per_dim_2 = [k_max, qmax_bin]
tpb2 = [1, 64]
bpg2 = generate_grid(elements_per_dim_2, tpb2)

elements_per_dim_nq = [i_max - i_min, qmax_bin]
tpbnq = [1, 64]
bpgnq = generate_grid(elements_per_dim_nq, tpbnq)
qbin = .1
data1 = {'q': q, 'k_max': k_max, 'offset': offset, 'bpg1': bpg1, 'tpb1': tpb1,
         'bpg2': bpg2, 'tpb2': tpb2, 'scatter_array':scatter_array, 'qbin':qbin}


@ddt
class TestFlatKernel:
    @unpack
    @data(data1)
    def test_d(self, **kwargs):
        cda = np.zeros((k_max, 3), np.float32)
        cd(cda, q, offset)

        with gpu:
            gda = np.zeros((k_max, 3), np.float32)
            dgda = cuda.to_device(gda)
            dq = cuda.to_device(q)
            gd[bpg1, tpb1](dgda, dq, offset)
            dgda.to_host()

        assert_allclose(gda, cda)

    @unpack
    @data(data1)
    def test_r(self, **kwargs):
        d = get_d(**kwargs)
        cra = np.zeros(k_max, np.float32)
        cr(cra, d)

        with gpu:
            gra = np.zeros(k_max, np.float32)
            dgra = cuda.to_device(gra)
            dd = cuda.to_device(d)
            gr[bpg1, tpb1](dgra, dd)
            dgra.to_host()

        assert_allclose(gra, cra)

    @unpack
    @data(data1)
    def test_norm(self, **kwargs):
        cnorm = np.zeros((k_max, qmax_bin), np.float32)
        cn(cnorm, scatter_array, k_cov)

        with gpu:
            gnorm = np.zeros((k_max, qmax_bin), np.float32)
            dgnorm = cuda.to_device(gnorm)
            dscat = cuda.to_device(scatter_array)
            gn[bpg2, tpb2](dgnorm, dscat, k_cov)
            dgnorm.to_host()

        assert_allclose(gnorm, cnorm)

    @unpack
    @data(data1)
    def test_norm(self, **kwargs):
        cnorm = np.zeros((k_max, qmax_bin), np.float32)
        cn(cnorm, scatter_array, k_cov)

        with gpu:
            gnorm = np.zeros((k_max, qmax_bin), np.float32)
            dgnorm = cuda.to_device(gnorm)
            dscat = cuda.to_device(scatter_array)
            gn[bpg2, tpb2](dgnorm, dscat, k_cov)
            dgnorm.to_host()

        assert_allclose(gnorm, cnorm)

    @unpack
    @data(data1)
    def test_fq(self, **kwargs):
        r = get_r(**kwargs).astype(np.float32)
        norm = get_norm(**kwargs).astype(np.float32)

        cfqa = np.zeros((k_max, qmax_bin), np.float32)
        cfq(cfqa, r, norm, qbin)

        with gpu:
            gfqa = np.zeros((k_max, qmax_bin), np.float32)
            dgfqa = cuda.to_device(gfqa)
            dr = cuda.to_device(r)
            dnorm = cuda.to_device(norm)
            gfq[bpg2, tpb2](dgfqa, dr, dnorm, qbin)
            dgfqa.to_host()
        assert_allclose(gfqa, cfqa)


def get_d(**kwargs):
    cda = np.zeros((k_max, 3), np.float32)
    cd(cda, q, offset)
    return cda


def get_r(**kwargs):
    d = get_d(**kwargs)
    cra = np.zeros(k_max, np.float32)
    cr(cra, d)
    return cra

def get_norm(**kwargs):
    cnorm = np.zeros((k_max, qmax_bin), np.float32)
    cn(cnorm, scatter_array, k_cov)
    return cnorm

def get_fq(**kwargs):
    r = get_r(**kwargs).astype(np.float32)
    norm = get_norm(**kwargs).astype(np.float32)

    cfqa = np.zeros((k_max, qmax_bin), np.float32)
    cfq(cfqa, r, norm, qbin)

    with gpu:
        gfqa = np.zeros((k_max, qmax_bin), np.float32)
        dgfqa = cuda.to_device(gfqa)
        dr = cuda.to_device(r)
        dnorm = cuda.to_device(norm)
        gfq[bpg2, tpb2](dgfqa, dr, dnorm, qbin)
        dgfqa.to_host()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1)
    # i1 = ax.imshow(gfqa, aspect='auto', interpolation='None')
    # plt.colorbar(i1)

    # ax2 = fig.add_subplot(1, 3, 2)
    # i2 = ax2.imshow(cfqa, aspect='auto', interpolation='None')
    # plt.colorbar(i2)
    print np.amax(np.abs(gfqa - cfqa)), np.std(gfqa - cfqa)
    plt.imshow(gfqa - cfqa, aspect='auto', interpolation='None')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # import nose
    #
    # nose.runmodule(argv=['-s', '--with-doctest', '-v'], exit=False)
    get_fq()