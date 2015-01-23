__author__ = 'christopher'
import matplotlib.pyplot as plt
from timeit import default_timer as time
import numpy as np
from numbapro import cuda
from pyiid.gpu.numbapro_cuda_kernels import *

def wrap_fq_gpu(atoms, qmax=25., qbin=.1):
    #get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    #build the empty arrays
    d = np.zeros((n, n, 3), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)

    #cuda kernel information

    stream = cuda.stream()
    tpb = 32
    bpg = int(math.ceil(float(n) / tpb))


    #start calculations
    s = time()

    dscat = cuda.to_device(scatter_array, stream)
    dnorm = cuda.to_device(norm_array)
    get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)

    dd = cuda.to_device(d, stream)
    dq = cuda.to_device(q, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)

    get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
    cuda.synchronize()

    get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)

    cuda.synchronize()
    dr.to_host(stream)
    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
    cuda.synchronize()


    get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
    cuda.synchronize()
    dnorm.to_host()

    get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
    cuda.synchronize()
    get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
    cuda.synchronize()
    dfq.to_host(stream)
    fq = super_fq.sum(axis=(0, 1))
    na = norm_array.sum(axis=(0, 1))

    na *= 1. / (scatter_array.shape[0] ** 2)
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * na) * fq)
    np.seterr(**old_settings)
    # print(time()-s)
    return fq


def wrap_fq_grad_gpu(atoms, qmax=25., qbin=.1):
    #atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    #build empty arrays
    d = np.zeros((n, n, 3), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
    tzr = np.zeros((n, n, qmax_bin), dtype=np.float32)
    super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)

    #cuda info
    stream = cuda.stream()
    tpb = 32
    bpg = int(math.ceil(float(n) / tpb))

    #start calculations
    s = time()
    dscat = cuda.to_device(scatter_array, stream)
    dnorm = cuda.to_device(norm_array)
    get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)

    dd = cuda.to_device(d, stream)
    dq = cuda.to_device(q, stream)
    dr = cuda.to_device(r, stream)
    dfq = cuda.to_device(super_fq, stream)

    get_d_array[(bpg, bpg), (tpb, tpb), stream](dd, dq)
    cuda.synchronize()

    get_r_array[(bpg, bpg), (tpb, tpb), stream](dr, dd)

    cuda.synchronize()
    dr.to_host(stream)
    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dfq, dr, qbin)
    cuda.synchronize()

    cuda.synchronize()

    get_fq_p1[(bpg, bpg), (tpb, tpb), stream](dfq)
    cuda.synchronize()

    get_fq_p2[(bpg, bpg), (tpb, tpb), stream](dfq, dr)
    cuda.synchronize()
    get_fq_p3[(bpg, bpg), (tpb, tpb), stream](dfq, dnorm)
    cuda.synchronize()
    dfq.to_host(stream)

    rgrad = np.zeros((n, n, 3), dtype=np.float32)
    q_over_r = np.zeros((n, n, qmax_bin), dtype=np.float32)
    cos_term = np.zeros((n, n, qmax_bin), dtype=np.float32)
    grad_p = np.zeros((n, n, 3, qmax_bin), dtype=np.float32)
    kqr = np.zeros((n, n, qmax_bin), dtype=np.float32)

    drgrad = cuda.to_device(rgrad, stream)
    dq_over_r = cuda.to_device(q_over_r, stream)
    dcos_term = cuda.to_device(cos_term, stream)
    dgrad_p = cuda.to_device(grad_p, stream)
    dkqr = cuda.to_device(kqr, stream)


    get_fq_p0[(bpg, bpg), (tpb, tpb), stream](dkqr, dr, qbin)
    cuda.synchronize()

    fq_grad_position1[(bpg, bpg), (tpb, tpb), stream](dq_over_r, dr, qbin) #OK
    cuda.synchronize()



    fq_grad_position0[(bpg, bpg), (tpb, tpb), stream](drgrad, dd, dr) #OK
    fq_grad_position3[(bpg, bpg), (tpb, tpb), stream](dcos_term, dkqr) #OK
    cuda.synchronize()
    drgrad.to_host(stream)
    dcos_term.to_host(stream)
    dkqr.to_host(stream)
    print 'gpu kqr', kqr[1,2,5]
    print'p3, gpu', cos_term[1,2,5]

    fq_grad_position4[(bpg, bpg), (tpb, tpb), stream](dcos_term, dq_over_r) #OK
    cuda.synchronize()

    fq_grad_position5[(bpg, bpg), (tpb, tpb), stream](dcos_term, dnorm) #OK
    cuda.synchronize()

    fq_grad_position6[(bpg, bpg), (tpb, tpb), stream](dfq, dr) #OK
    cuda.synchronize()

    fq_grad_position7[(bpg, bpg), (tpb, tpb), stream](dcos_term, dfq) #OK
    cuda.synchronize()

    fq_grad_position_final1[(bpg, bpg), (tpb, tpb), stream](dgrad_p, drgrad) #OK
    cuda.synchronize()

    fq_grad_position_final2[(bpg, bpg), (tpb, tpb), stream](dgrad_p, dcos_term) #ok
    dgrad_p.to_host(stream)


    #sum down to 1D array
    grad_p=grad_p.sum(axis=(1))
    # print grad_p
    dnorm.to_host(stream)

    #sum reduce to 1D
    na = norm_array.sum(axis=(0, 1))
    na *= 1. / (scatter_array.shape[0] ** 2)

    old_settings = np.seterr(all='ignore')
    # print na
    for tx in range(n):
            for tz in range(3):
                # grad_p[tx, tz, :qmin_bin] = 0.0
                grad_p[tx, tz] = np.nan_to_num(
                    1 / (n * na) * grad_p[tx, tz])
    np.seterr(**old_settings)
    # print(time()-s)
    return grad_p
