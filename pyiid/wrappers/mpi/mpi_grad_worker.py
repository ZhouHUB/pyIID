__author__ = 'christopher'
from mpi4py import MPI
from numba import cuda
import numpy as np
import math
import sys
if __name__ == '__main__':

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    total_data = None
    n_cov_list = []
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):

        q, scatter_array, qmax_bin, qbin, m, n_cov = task
        gpus = cuda.gpus.lst
        gpu = gpus[0]
        # Build Data arrays
        n = len(q)
        tups = [(m, n, 3), (m, n),
                (m, n, qmax_bin), (m, n, qmax_bin),
                (m, n, 3, qmax_bin), (m, n, qmax_bin)]
        data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
        if total_data is None:
            total_data = np.zeros(data[4].shape)
        with gpu:
            sys.path.extend(['/mnt/work-data/dev/pyIID'])
            from pyiid.kernels.multi_cuda import get_normalization_array, \
                get_d_array, get_r_array, get_fq_p0_1_2, get_fq_p3, \
                fq_grad_position3, fq_grad_position5, fq_grad_position7, \
                fq_grad_position_final1, fq_grad_position_final2
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
            dnorm = cuda.to_device(data[2], stream2)

            '--------------------------------------------------------------'
            get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)
            '--------------------------------------------------------------'
            dd = cuda.to_device(data[0], stream)
            dr = cuda.to_device(data[1], stream)
            dfq = cuda.to_device(data[3], stream)
            dq = cuda.to_device(q, stream)

            get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
            # cuda.synchronize()

            get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

            '--------------------------------------------------------------'
            get_fq_p0_1_2[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
            '--------------------------------------------------------------'
            dcos_term = cuda.to_device(data[5], stream2)
            # cuda.synchronize()


            get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
            fq_grad_position3[bpg_l_3, tpb_l_3, stream3](dcos_term, dr, qbin)
            dgrad_p = cuda.to_device(data[4], stream2)
            # cuda.synchronize()


            # cuda.synchronize()

            fq_grad_position_final1[bpg_l_3, tpb_l_3, stream](dgrad_p, dd, dr)
            fq_grad_position5[bpg_l_3, tpb_l_3, stream2](dcos_term, dnorm)
            # cuda.synchronize()

            fq_grad_position7[bpg_l_3, tpb_l_3, stream](dcos_term, dfq, dr)
            # cuda.synchronize()

            fq_grad_position_final2[bpg_l_3, tpb_l_3, stream](dgrad_p, dcos_term)
            dgrad_p.to_host(stream)
            total_data += data[4]
            n_cov_list.append(n_cov)
            del data, dscat, dnorm, dd, dr, dfq, dcos_term, dgrad_p

    final_data = total_data.sum(axis=1)
    return_msg = (n_cov, final_data)
    #Return Finished Data
    comm.gather(sendobj=return_msg, root=0)
    # Shutdown
    comm.Disconnect()
