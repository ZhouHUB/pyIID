__author__ = 'christopher'
from pyiid.kernels.cpu_kernel import *
import numpy as np
from numbapro import autojit


def wrap_fq(atoms, qbin=.1):
    """
    Generate the reduced structure function

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment

    Returns
    -------
    
    fq:1darray
        The reduced structure function
    """
    q = atoms.get_positions()

    # get scatter array
    scatter_array = atoms.get_array('scatter')
    # define scatter_q information and initialize constants

    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)


    # get non-normalized fq
    fq = np.zeros(qmax_bin)
    get_fq_array(fq, r, scatter_array, qbin)


    #Normalize fq
    # norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
    # get_normalization_array(norm_array, scatter_array)

    # norm_array = norm_array.sum(axis=(0, 1))
    # norm_array *= 1. / (scatter_array.shape[0] ** 2)

    # old_settings = np.seterr(all='ignore')
    # fq = np.nan_to_num(1 / (n * norm_array) * fq)
    # np.seterr(**old_settings)
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    # del norm_array, r, d, q, scatter_array
    del r, d, q, scatter_array
    return fq


def wrap_fq_grad(atoms, qbin=.1):
    """
    Generate the reduced structure function gradient

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment

    Returns
    -------
    
    dfq_dq:ndarray
        The reduced structure function gradient
    """
    q = atoms.get_positions()

    # get scatter array
    scatter_array = atoms.get_array('scatter')

    # define scatter_q information and initialize constants
    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # get non-normalized FQ

    #Normalize FQ
    norm_array = np.zeros((n, n, qmax_bin))
    get_normalization_array(norm_array, scatter_array)
    norm_array = norm_array.sum(axis=(0, 1))
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

    dfq_dq = np.zeros((n, 3, qmax_bin))
    fq_grad_position(dfq_dq, d, r, scatter_array, qbin)
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            dfq_dq[tx, tz] = np.nan_to_num(
                1 / (n * norm_array) * dfq_dq[tx, tz])
    np.seterr(**old_settings)
    del q, d, r, scatter_array, norm_array
    return dfq_dq


def spring_nrg(atoms, k, rt):
    q = atoms.positions
    n = len(atoms)
    d = np.zeros((n, n, 3))
    get_d_array(d, q)
    r = np.zeros((n, n))
    get_r_array(r, d)

    thresh = np.less(r, rt)
    for i in range(len(thresh)):
        thresh[i, i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh]-rt)

    energy = np.sum(mag[thresh]/2.*(r[thresh]-rt))
    return energy


def spring_force(atoms, k, rt):
    q = atoms.positions
    n = len(atoms)
    d = np.zeros((n, n, 3))
    get_d_array(d, q)
    r = np.zeros((n, n))
    get_r_array(r, d)

    thresh = np.less(r, rt)
    for i in range(len(thresh)):
        thresh[i,i] = False

    mag = np.zeros(r.shape)
    mag[thresh] = k * (r[thresh]-rt)

    direction = np.zeros((n, n, 3))
    old_settings = np.seterr(all='ignore')
    for tz in range(3):
        direction[thresh, tz] = d[thresh, tz]/r[thresh] * mag[thresh]
    np.seterr(**old_settings)
    direction[np.isnan(direction)] = 0.0
    direction = np.sum(direction, axis=1)
    return direction
