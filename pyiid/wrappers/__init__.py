import math
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
        scatter_array = atoms.get_array('F(Q) scatter').astype(np.float32)
    else:
        scatter_array = atoms.get_array('PDF scatter').astype(np.float32)
    qmax_bin = scatter_array.shape[1]
    sort_gpus, sort_gmem = get_gpus_mem()

    return q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem


def generate_grid(elements, tpb):
    assert len(elements) == len(tpb)
    bpgs = []
    for e_dim, thds in zip(elements, tpb):
        bpg = int(math.ceil(float(e_dim) / thds))
        if bpg < 1:
            bpg = 1
        bpgs.append(bpg)
    for e, t, b in zip(elements, tpb, bpgs):
        # print e, t*b
        assert (e <= t * b)
    return bpgs