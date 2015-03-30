__author__ = 'christopher'

from numba import cuda
import numpy as np

# Check GPU memory via MPI
def get_memory():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    print rank

    gpus = cuda.gpus.lst
    mem = np.zeros(len(gpus), dtype=np.int)
    for i, gpu in enumerate(gpus):
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem[i] = (meminfo[0])
    comm.gather(mem, root=0)

    if rank == 0:
        return mem
if __name__ == '__main__':
    mem = get_memory()