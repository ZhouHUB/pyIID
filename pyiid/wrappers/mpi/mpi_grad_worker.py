__author__ = 'christopher'
if __name__ == '__main__':
    import numpy as np
    import math
    import sys
    from mpi4py import MPI
    from numba import cuda

    from ..nxn_atomic_gpu import atomic_grad_fq

    grad_cov = []
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):
        gpus = cuda.gpus.lst
        gpu = gpus[0]
        grad_cov.append(atomic_grad_fq(*task))
        cuda.close()
    # Return Finished Data
    comm.gather(sendobj=grad_cov, root=0)
    # Shutdown
    comm.Disconnect()