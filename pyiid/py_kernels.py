__author__ = 'christopher'

import numpy as np
import math
from numbapro import cuda


@jit(argtypes=[float32[:, :, 3], float32[:, 3]], target='gpu')
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    D: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    tx, ty, tz = cuda.grid(3)
    d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


@jit(argtypes=[float32[:, 3], float32[:, :, 3]], target='gpu')
def get_r_array(r, d):
    """
    This kernel reduces the D matrix from NxNx3 to NxN by converting the
    coordinate wise distances to a total distance via x**2+y**2+z**2 =
    d**2.  The resulting array should have zero diagonals and be symmetric.
    """
    tx, ty = cuda.grid(2)
    r[tx, ty] = math.sqrt(
        d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty,
                                                  2] ** 2)


@jit(argtypes=[float32[:], float32[:, 3], float32[:, :], float32], target='gpu')
def get_fq_array(fq, r, scatter_array, Qbin):
    """
    Generate F(Q), not normalized, via the debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    n_range: Nd array
        The range of number of atoms
    Qmax_Qmin_bin_range:
        Range between Qmin and Qmax
    Qbin:
        The Qbin size
    """
    smscale = 1
    tx, ty, kq = cuda.grid(3)
    if tx != ty:
        dwscale = 1
    fq[kq] += smscale * \
              dwscale * \
              scatter_array[tx, kq] * \
              scatter_array[ty, kq] / \
              r[tx, ty] * \
              math.sin(kq * Qbin * r[tx, ty])