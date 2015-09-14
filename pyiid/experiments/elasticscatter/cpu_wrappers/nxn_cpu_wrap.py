import numpy as np

from pyiid.experiments.elasticscatter.kernels.cpu_nxn import *

__author__ = 'christopher'


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
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
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    # define scatter_q information and initialize constants

    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin))
    get_normalization_array(norm, scatter_array)

    # Get omega array
    omega = np.zeros((n, n, qmax_bin))
    get_omega(omega, r, qbin)
    
    if hasattr(atoms, 'adp'):
        adps = atoms.adp
        sigma = np.zeros(n, n, np.float32)
        get_sigma_from_adp(sigma, adps, r, d)

        tau = np.zeros((n, n, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        fq = np.zeros((n, n, qmax_bin), np.float32)
        get_adp_fq(fq, omega, tau, norm)
        del tau, sigma, adps
    else:
        # get non-normalized fq
        fq = np.zeros(qmax_bin)
        get_fq(fq, omega, norm)

    # Normalize fq
    na = np.mean(norm, axis=(0, 1)) * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(fq / na)
    np.seterr(**old_settings)
    del q, d, r, norm, omega, na
    return fq


def wrap_apd_fq(atoms, qbin=.1, sum_type='fq'):
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
    adps = atoms.adps.get_position()

    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    # define scatter_q information and initialize constants

    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)
    del q

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    norm = np.zeros((n, n, qmax_bin))
    get_normalization_array(norm, scatter_array)

    sigma = np.zeros((n, n))
    get_sigma_from_adp(sigma, adps, r, d)

    tau = np.zeros((n, n, qmax_bin))
    get_tau(tau, sigma, qbin)

    omega = np.zeros((n, n, qmax_bin))
    get_omega(omega, r, qbin)

    # get non-normalized fq
    fq = np.zeros(qmax_bin)
    get_adp_fq(fq, omega, tau, norm)

    # Normalize fq
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    del r, scatter_array, na, d
    return fq


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
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
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')

    # define scatter_q information and initialize constants
    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin))
    get_normalization_array(norm, scatter_array)

    # Get omega
    omega = np.zeros((n, n, qmax_bin))
    get_omega(omega, r, qbin)

    # Get grad omega
    grad_omega = np.zeros((n, n, 3, qmax_bin))
    get_grad_omega(grad_omega, omega, r, d, qbin)
    
    if hasattr(atoms, 'adp'):
        adps = atoms.adp
        sigma = np.zeros(n, n, np.float32)
        get_sigma_from_adp(sigma, adps, r, d)

        tau = np.zeros((n, n, qmax_bin), np.float32)
        get_tau(tau, sigma, qbin)

        grad_tau = np.zeros((n, n, 3, qmax_bin))
        get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin, k_cov)

        grad = np.empty((n, n, 3, qmax_bin), np.float32)
        get_adp_grad_fq(grad, omega, tau, grad_omega, grad_tau, norm)
        del tau, sigma, adps
    else:
        # Get grad FQ
        grad_fq = np.zeros((n, 3, qmax_bin))
        get_grad_fq(grad_fq, grad_omega, norm)

    # Normalize FQ
    na = np.mean(norm, axis=(0, 1)) * n
    old_settings = np.seterr(all='ignore')
    grad_fq = np.nan_to_num(grad_fq / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega, grad_omega
    return grad_fq


def wrap_apd_fq_grad(atoms, qbin=.1, sum_type='fq'):
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
    adps = atoms.adps.get_position()

    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')

    # define scatter_q information and initialize constants
    qmax_bin = scatter_array.shape[1]
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d)

    # Get normalization array
    norm = np.zeros((n, n, qmax_bin))
    get_normalization_array(norm, scatter_array)

    # Get omega
    omega = np.zeros((n, n, qmax_bin))
    get_omega(omega, r, qbin)

    sigma = np.zeros((n, n))
    get_sigma_from_adp(sigma, adps, r, d)

    tau = np.zeros((n, n, qmax_bin))
    get_tau(tau, sigma, qbin)

    # Get grad omega
    grad_omega = np.zeros((n, n, 3, qmax_bin))
    get_grad_omega(grad_omega, omega, r, d, qbin)

    grad_tau = np.zeros((n, n, 3, qmax_bin))
    get_grad_tau(grad_tau, tau, r, d, sigma, adps, qbin)

    # Get grad FQ
    grad_fq = np.zeros((n, 3, qmax_bin))
    get_adp_grad_fq(grad_fq, omega, tau, grad_omega, grad_tau, norm)

    # Normalize FQ
    na = np.mean(norm, axis=(0, 1)) * n
    old_settings = np.seterr(all='ignore')
    grad_fq = np.nan_to_num(grad_fq / na)
    np.seterr(**old_settings)
    del d, r, scatter_array, norm, omega, grad_omega
    return grad_fq
