import math

__author__ = 'christopher'


def ij_to_k(i, j):
    return int(j + i * (i - 1) / 2)


def k_to_ij(k):
    i = math.floor(float((1 + math.sqrt(1 + 8. * k))) / 2.)
    j = k - i * (i - 1) / 2.
    return int(i), int(j)