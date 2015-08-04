__author__ = 'christopher'
from numba import *
from flat_kernel import cuda_k_to_ij

'''
@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def get_normalization_array(norm_array, scat, offset):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNx3 array
        Normalization array
    scatter_array: NxM array
        The scatter factor array
    """
    # snorm = cuda.shared.array((1, 64), f4)
    snormi = cuda.shared.array((1, 64), f4)
    snormj = cuda.shared.array((1, 64), f4)
    k, qx = cuda.grid(2)
    n = norm_array.shape[0]
    qmax_bin = norm_array.shape[1]
    if k >= n or qx >= qmax_bin:
        return
    tid = cuda.threadIdx.y
    i, j = cuda_k_to_ij(int32(k + offset))
    snormi[0, tid] = scat[i, qx]
    snormj[0, tid] = scat[j, qx]
    cuda.syncthreads()

    snormi[0, tid] *= snormj[0, tid]
    cuda.syncthreads()
    # norm_array[k, qx] = scat[i, qx] * scat[j, qx]
    norm_array[k, qx] = snormi[0, tid]


# A[k, q] = norm*Q, B[k, q] = cos(Q*r), C[k, w] = d/r/r
# D[k, q] = A*B - F(Q)
# E[k, w, q] = D * C
@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4])
def get_grad_fq_a(A, norm, qbin):
    k, qx = cuda.grid(2)
    if k >= len(A) or qx >= A.shape[1]:
        return
    A[k, qx] = norm[k, qx] * float32(qx * qbin)


@cuda.jit(argtypes=[f4[:, :], f4[:], f4])
def get_grad_fq_b(B, r, qbin):
    k, qx = cuda.grid(2)
    if k >= len(B) or qx >= B.shape[1]:
        return
    B[k, qx] = math.cos(float32(qx * qbin) * r[k])


@cuda.jit(argtypes=[f4[:], f4[:, :]])
def get_grad_fq_c(r, d):
    k = cuda.grid(1)
    if k >= len(r):
        return
    for w in range(3):
        d[k, w] /= r[k] ** 2


# @cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :], f4[:, :]])
@cuda.jit(argtypes=[f4[:, :], f4[:, :], f4[:, :]])
def get_grad_fq_d(A, B, fq):
    k, qx = cuda.grid(2)
    if k >= len(A) or qx >= A.shape[1]:
        return
    # D[k, qx] = A[k, qx] * B[k, qx] - fq[k, qx]
    A[k, qx] *= B[k, qx]
    A[k, qx] -= fq[k, qx]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4[:, :]])
def get_grad_fq_e(E, D, C):
    k, qx = cuda.grid(2)
    if k >= len(E) or qx >= E.shape[2]:
        return
    for w in range(3):
        E[k, w, qx] = D[k, qx] * C[k, w]
"""
'''

@cuda.jit(argtypes=[f4[:, :], f4[:, :], i4])
def experimental_sum_fq(g_odata, g_idata, n):
    _, qx = cuda.grid(2)
    sdata = cuda.shared.array(512, f4)

    tid = cuda.threadIdx.x
    bd = cuda.blockDim.x
    bid = cuda.blockIdx.x
    i = bid * bd * 2 + tid
    gridSize = bd * 2 * cuda.gridDim.x

    sdata[tid] = 0.
    while i < n:
        if i + bd >= len(g_idata):
            sdata[tid] += g_idata[i, qx]
        else:
            sdata[tid] += g_idata[i, qx] + g_idata[i + bd, qx]
        i += gridSize
    cuda.syncthreads()

    if bd >= 512:
        if tid < 256:
            sdata[tid] += sdata[tid + 256]
        cuda.syncthreads()

    if bd >= 256:
        if tid < 128:
            sdata[tid] += sdata[tid + 128]
        cuda.syncthreads()

    if bd >= 128:
        if tid < 64:
            sdata[tid] += sdata[tid + 64]
        cuda.syncthreads()

    if tid < 32:
        if bd >= 64:
            sdata[tid] += sdata[tid + 32]
        if bd >= 32:
            sdata[tid] += sdata[tid + 16]
        if bd >= 16:
            sdata[tid] += sdata[tid + 8]
        if bd >= 8:
            sdata[tid] += sdata[tid + 4]
        if bd >= 4:
            sdata[tid] += sdata[tid + 2]
        if bd >= 2:
            sdata[tid] += sdata[tid + 1]

    if tid == 0:
        g_odata[cuda.blockIdx.x, qx] = sdata[0]

'''
# TODO: may need a closure here for the bpgn
@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], i4])
def experimental_sum_grad_fq(g_idata, g_odata, n):
    sdata = cuda.shared.array(bpgn, f4)
    i, _, qx = cuda.grid(2)
    tid = cuda.threadIdx.y
    bd = cuda.blockDim.y
    bid = cuda.blockIdx.y
    i = bid * bd * 2 + tid
    gridSize = bd * 2 * cuda.gridDim.y

    sdata[tid] = 0
    while i < n:
        # perform the mapping here
        sdata[tid] += g_idata[i, qx] + g_idata[i + bd, qx]
        i += gridSize

    cuda.syncthreads()

    if bd >= 512:
        if tid < 256:
            sdata[tid] += sdata[tid + 256]
        cuda.syncthreads()

    if bd >= 256:
        if tid < 128:
            sdata[tid] += sdata[tid + 128]
        cuda.syncthreads()

    if bd >= 128:
        if tid < 64:
            sdata[tid] += sdata[tid + 64]
        cuda.syncthreads()

    if tid < 32:
        sdata[tid] += sdata[tid + 32]
        sdata[tid] += sdata[tid + 16]
        sdata[tid] += sdata[tid + 8]
        sdata[tid] += sdata[tid + 4]
        sdata[tid] += sdata[tid + 2]
        sdata[tid] += sdata[tid + 1]

    if tid == 0:
        g_odata[i, cuda.blockIdx.y, qx] = sdata[0]'''

@cuda.jit(argtypes=[f4[:], f4[:, :]])
def d2_to_d1_cleanup_kernel(out_data, in_data):
    i = cuda.grid(1)
    if i >= len(out_data):
        return
    out_data[i] = in_data[0, i]

@cuda.jit(argtypes=[f4[:, :]])
def d2_zero(A):
    i, j = cuda.grid(2)
    if i >= A.shape[0] or j >= A.shape[1]:
        return
    A[i, j] = f4(0.)