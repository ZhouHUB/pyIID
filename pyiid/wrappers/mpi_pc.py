__author__ = 'christopher'


# PARENT
def wrap_fq(atoms, qmax=25., qbin=.1):

    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    # Get GPU information
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]

    fq_q = []
    norm_q = []
    n_cov = 0
    p_dict = {}
    while n_cov < n:
        for gpu, mem in zip(sort_gpus, sort_gmem):
            m = int(math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                8 * n * (qmax_bin + 2))))
            if m > n - n_cov:
                m = n - n_cov

            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if n_cov >= n:
                    break
                p = Thread(
                    target=sub_fq, args=(
                        gpu, q, scatter_array,
                        fq_q, norm_q, qmax_bin, qbin, m, n_cov))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov >= n:
                    break

    for value in p_dict.values():
        value.join()

    fq = np.zeros(qmax_bin)
    na = np.zeros(qmax_bin)
    for ele, ele2 in zip(fq_q, norm_q):
        fq[:] += ele
        na[:] += ele2
    na *= 1. / (scatter_array.shape[0] ** 2)
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * na) * fq)
    np.seterr(**old_settings)
    return fq


# Children

# Check GPU memory via MPI
def get_memory():
