from numba import cuda
import numpy as np

__author__ = 'christopher'


def get_gpus_mem():
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(int(meminfo[0]))
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    return sort_gpus, sort_gmem

def setup_gpu_calc(atoms, sum_type):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    qmax_bin = scatter_array.shape[1]
    sort_gpus, sort_gmem = get_gpus_mem()

    return q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem