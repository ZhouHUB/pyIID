__author__ = 'christopher'

import numpy as np
import math
from numbapro import cuda


@jit(argtypes=[float32[:,:,3], float32[:, 3]], target='gpu')
def generate_d(d, q):
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

@jit(argtypes=[float32[:,:,3], float32[:, 3]], target='gpu')
def generate_d(d, q):
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

@jit(argtypes=[float32[:, 3], float32[:,:,3]], target='gpu')
def reduce_d(reduced_d, d):
    """
    This kernel reduces the D matrix from NxNx3 to NxN by converting the
    coordinate wise distances to a total distance via x**2+y**2+z**2 =
    d**2.  The resulting array should have zero diagonals and be symmetric.
    """
    tx, ty = cuda.grid(2)
    reduced_d[tx, ty] = math.sqrt(d[tx, ty, 0]**2+d[tx, ty, 1]**2+d[tx, ty,
                                                                   2]**2)

