__author__ = 'christopher'
if __name__ == '__main__':
    import numpy as np
    from mpi4py import MPI
    from numba import cuda

    from pyiid.wrappers.gpu_wrappers.nxn_atomic_gpu import atomic_fq

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    total_data = []
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):

        # with gpu?
        # unpack the message sent from the head process
        q, scatter_array, qbin, m, n_cov = task
        gpus = cuda.gpus.lst
        gpu = gpus[0]
        gpu_return = atomic_fq(*task)
        total_data.append(gpu_return)

        cuda.close()
    final_data = np.asarray(total_data)
    comm.gather(sendobj=final_data.sum(axis=0), root=0)

    # Shutdown process
    comm.Disconnect()
