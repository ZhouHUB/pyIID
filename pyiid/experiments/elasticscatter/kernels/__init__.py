import math
from numba import *

__author__ = 'christopher'


@jit(target='cpu', nopython=True)
def ij_to_k(i, j):
    return int(j + i * (i - 1) / 2)


@jit(target='cpu', nopython=True)
def k_to_ij(k):
    i = math.floor(float((1 + math.sqrt(1 + 8. * k))) / 2.)
    j = k - i * (i - 1) / 2.
    return int(i), int(j)
