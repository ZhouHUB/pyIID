from numba import *

from pyiid.kernels import *
__author__ = 'christopher'

processor_target = 'cpu'

# F(Q) test_kernels -----------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_d_array(d, q, offset):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    for k in range(len(d)):
        i = int(math.floor(float((1 + math.sqrt(1 + 8. * (k + offset)))) / 2.))
        j = int((k + offset) - i * (i - 1) / 2.)
        for tz in range(3):
            d[k, tz] = q[i, tz] - q[j, tz]


@jit(target=processor_target, nopython=True)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    for k in xrange(len(r)):
        r[k] = math.sqrt(d[k, 0] ** 2 + d[k, 1] ** 2 + d[k, 2] ** 2)


@jit(target=processor_target, nopython=True)
def get_normalization_array(norm, scat, offset):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    for k in xrange(norm.shape[0]):
        i = int(math.floor(float((1 + math.sqrt(1 + 8. * (k + offset)))) / 2.))
        j = int((k + offset) - i * (i - 1) / 2.)
        for qx in xrange(norm.shape[1]):
            norm[k, qx] = scat[i, qx] * scat[j, qx]


@jit(target=processor_target, nopython=True)
def get_fq(fq, r, norm, qbin):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    for k in xrange(fq.shape[0]):
        for qx in xrange(fq.shape[1]):
            Q = float32(qbin) * float32(qx)
            # Q = float32(qbin * qx)
            rk = r[k]
            fq[k, qx] = norm[k, qx] * math.sin(Q * rk) / rk


# Gradient test_kernels -------------------------------------------------------
@jit(target=processor_target, nopython=True)
def get_grad_fq(grad, fq, r, d, norm, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    for k in xrange(grad.shape[0]):
        for qx in xrange(fq.shape[1]):
            # Q = float32(qbin * qx)
            Q = float32(qbin) * float32(qx)
            rk = r[k]
            A = (norm[k, qx] * Q * math.cos(Q * rk) - fq[k, qx]) / float32(
                rk ** 2)
            for w in range(3):
                grad[k, w, qx] = A * d[k, w]
                # grad[k, w, qx] = float32(qbin)


@jit(target=processor_target, nopython=True)
def fast_fast_flat_sum(new_grad, grad, k_cov):
    n = len(new_grad)
    k_max = len(grad)
    for i in xrange(n):
        for j in xrange(n):
            for qx in xrange(grad.shape[2]):
                for tz in xrange(3):
                    if j < i:
                        k = j + i * (i - 1) / 2 - k_cov
                        alpha = -1
                    elif i < j:
                        k = i + j * (j - 1) / 2 - k_cov
                        alpha = 1
                    if 0 <= k < k_max:
                        new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
                        # new_grad[i, tz, qx] += alpha


'''
    n = len(new_grad)
    k_max = len(grad)
    for i in xrange(n):
        for j in xrange(n):
            for qx in xrange(grad.shape[2]):
                k = -1
                alpha = 0
                if j < i:
                    k = j + i * (i - 1) / 2
                    alpha = float32(1.)
                    k -= k_cov
                elif i < j:
                    k = i + j * (j - 1) / 2
                    alpha = float32(-1)
                    k -= k_cov
                # if 0 <= k <= k_max:
                if 0 <= k < k_max:
                    for tz in range(3):
                        # new_grad[i, tz, qx] += grad[k, tz, qx] * alpha
                        new_grad[i, tz, qx] += alpha
    '''
'''
# Misc. Kernels----------------------------------------------------------------

@jit(target=targ, nopython=True)
def get_dw_sigma_squared(s, u, r, d, n):
    for tx in range(n):
        for ty in range(n):
            rnormx = d[tx, ty, 0] / r[tx, ty]
            rnormy = d[tx, ty, 1] / r[tx, ty]
            rnormz = d[tx, ty, 2] / r[tx, ty]
            ux = u[tx, 0] - u[ty, 0]
            uy = u[tx, 1] - u[ty, 1]
            uz = u[tx, 2] - u[ty, 2]
            u_dot_r = rnormx * ux + rnormy * uy + rnormz * uz
            s[tx, ty] = u_dot_r * u_dot_r


@jit(target=targ, nopython=True)
def get_gr(gr, r, rbin, n):
    """
    Generate gr the histogram of the atomic distances

    Parameters
    ----------
    gr: Nd array
    r: NxN array
    rbin: float
    n: Nd array
    :return:
    """
    for tx in range(n):
        for ty in range(n):
            gr[int(r[tx, ty] / rbin)] += 1


def simple_grad(grad_p, d, r):
    """
    Gradient of the delta function gr
    grad_p:
    d:
    r:
    :return:
    """
    n = len(r)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for tz in range(3):
                    grad_p[tx, tz] += d[tx, ty, tz] / (r[tx, ty] ** 3)


@jit(target=targ, nopython=True)
def spring_force_kernel(direction, d, r, mag):
    n = len(r)
    for i in range(n):
        for j in range(n):
            if i != j:
                direction[i, :] += d[i, j, :] / r[i, j] * mag[i, j]
'''
