__author__ = 'christopher'
import math


def get_d_array(d, q, n_range):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    r: NxNx3 array
    q: Nx3 array
        The atomic positions
    n_range: 1d array
        Range of atomic numbers
    """
    for tx in n_range:
        for ty in n_range:
            for tz in [0, 1, 2]:
                d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


def get_r_array(r, d, n_range):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    n_range: 1d array
        Range of atomic numbers
    """
    for tx in n_range:
        for ty in n_range:
            r[tx, ty] = math.sqrt(
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty,
                                                          2] ** 2)


def get_scatter_array(scatter_array, symbols, dpc, n_range,
                      Qmax_Qmin_bin_range, Qbin):
    """
    Generate the scattering array, which holds all the Q dependant scatter factors
    
    Parameters:
    ---------
    scatter_array: NxM array
        Holds the scatter factors
    symbols: Nd array
        Holds the string reps of the atomic symbols
    dpc: DebyePDFCalculator instance
        The gives method to get atomic scatter factors
    n_range:
        Number of atoms
    Qmax_Qmin_bin_range:
        Range between Qmin and Qmax
    Qbin:
        The Qbin size
    """
    for tx in n_range:
        for kq in Qmax_Qmin_bin_range:
            scatter_array[tx, kq] = dpc.scatteringfactortable.lookup(
                symbols[tx], q=kq * Qbin)


def get_fq_array(fq, r, scatter_array, n_range, Qmax_Qmin_bin_range, Qbin):
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
    for tx in n_range:
        for ty in n_range:
            if tx != ty:
                for kq in Qmax_Qmin_bin_range:
                    dwscale = 1
                    fq[kq] += smscale * dwscale * scatter_array[tx,
                                                                kq] * \
                              scatter_array[
                                  ty, kq] / r[tx, ty] * math.sin(
                        kq * Qbin * r[tx, ty])


def get_normalization_array(norm_array, scatter_array, Qmax_Qmin_bin_range,
                            n_range):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: Nd array
        Normalization array
    scatter_array: NxM array
        The scatter factor array
    Qmax_Qmin_bin_range:
        Range between Qmin and Qmax
     n_range: Nd array
        The range of number of atoms
    """
    for kq in Qmax_Qmin_bin_range:
        for tx in n_range:
            for ty in n_range:
                norm_array[kq] += (
                    scatter_array[tx, kq] * scatter_array[ty, kq])

                norm_array*1/(scatter_array.shape[0])**2

# def get_pdf_at_Qmin(qmin):