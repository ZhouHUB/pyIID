__author__ = 'christopher'
from mpi4py import MPI
from numba import cuda

comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

gpus = cuda.gpus.lst
mem_list = []
for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
sent_info = (rank, gpus, meminfo)
comm.gather(sendobj=sent_info, root=0)