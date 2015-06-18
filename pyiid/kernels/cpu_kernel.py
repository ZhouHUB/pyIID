__author__ = 'christopher'
import math
from numba import *
import mkl

targ = 'cpu'

# F(Q) test_kernels -----------------------------------------------------------

@autojit(target=targ)
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    n = len(q)
    for tx in range(n):
        for ty in range(n):
            for tz in range(3):
                d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


@autojit(target=targ)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    n = len(r)
    for tx in range(n):
        for ty in range(n):
            r[tx, ty] = math.sqrt(
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@autojit(target=targ)
def get_fq_array(fq, r, scatter_array, qbin):
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
    sum_scale = 1
    n = len(r)
    qmax_bin = len(fq)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    debye_waller_scale = 1
                    # TODO: debye_waller_scale = math.exp(
                    # -.5 * dw_signal_sqrd * (kq*Qbin)**2)
                    fq[kq] += sum_scale * \
                              debye_waller_scale * \
                              scatter_array[tx, kq] * \
                              scatter_array[ty, kq] / \
                              r[tx, ty] * \
                              math.sin(kq * qbin * r[tx, ty])


@autojit(target=targ)
def get_normalization_array(norm_array, scatter_array):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    n = len(norm_array)
    qmax_bin = norm_array.shape[2]

    for kq in range(0, qmax_bin):
        for tx in range(n):
            for ty in range(n):
                norm_array[tx, ty, kq] = (
                    scatter_array[tx, kq] * scatter_array[ty, kq])


# Gradient test_kernels -------------------------------------------------------
@autojit(target=targ)
def fq_grad_position(grad_p, d, r, scatter_array, qbin):
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
    n = len(r)
    qmax_bin = grad_p.shape[2]
    for tx in range(n):
        for tz in range(3):
            for ty in range(n):
                if tx != ty:
                    for kq in range(0, qmax_bin):
                        sub_grad_p = \
                            scatter_array[tx, kq] * \
                            scatter_array[ty, kq] * \
                            d[tx, ty, tz] * \
                            (
                                (kq * qbin) *
                                r[tx, ty] *
                                math.cos(kq * qbin * r[tx, ty]) -
                                math.sin(kq * qbin * r[tx, ty])
                            ) \
                            / (r[tx, ty] ** 3)
                        grad_p[tx, tz, kq] += sub_grad_p


# Misc. Kernels----------------------------------------------------------------

@autojit(target=targ)
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


@autojit(target=targ)
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


@autojit(target=targ)
def spring_force_kernel(direction, d, r, mag):
    n = len(r)
    for i in range(n):
        for j in range(n):
            if i != j:
                direction[i, :] += d[i, j, :] / r[i, j] * mag[i, j]
