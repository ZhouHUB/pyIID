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


def symmetric_reshape(out_data, in_data):
    for i in range(len(out_data)):
        for j in range(i):
            out_data[i, j] = in_data[j + i * (i - 1) / 2]
            out_data[j, i] = in_data[j + i * (i - 1) / 2]


def antisymmetric_reshape(out_data, in_data):
    for i in range(len(out_data)):
        for j in range(len(out_data)):
            if j < i:
                out_data[i, j] = -1 * in_data[ij_to_k(i, j)]
            elif i < j:
                out_data[i, j] = in_data[ij_to_k(j, i)]