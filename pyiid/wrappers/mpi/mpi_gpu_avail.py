
'''__author__ = 'christopher'
from numba import cuda

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
            with gpu:
                meminfo = cuda.current_context().get_memory_info()
            mem_list.append(int(meminfo[0]))
    # print type(rank), type(gpus), type(mem_list)
    cuda.close()
    comm.gather(sendobj=rank, root=0)
    comm.gather(sendobj=mem_list[0], root=0)
    comm.Disconnect()
   '''


__author__ = 'christopher'
from numba import cuda
import os

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    print os.getenv('HOSTNAME'), rank

    gpus = cuda.gpus.lst
    mem_list = []
    print gpus
    for gpu in gpus:
        print gpu, rank
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(int(meminfo[0]))
    # print type(rank), type(gpus), type(mem_list)
    cuda.close()
    comm.gather(sendobj=rank, root=0)
    comm.gather(sendobj=mem_list[0], root=0)
    comm.Disconnect()