__author__ = 'christopher'
from pyiid.serial_kernel import *
# from diffpy.srreal.pdfcalculator import DebyePDFCalculator
# TODO: Replace this SrFit dependence with scikit-xray
# dpc = DebyePDFCalculator()

def wrap_atoms(atoms, qmax=25., qmin=0.0, qbin=.1):
    """
    Call this function before applying calculator, it will generate static arrays for the scattering, preventing recalculation
    :param atoms:
    :param qmax:
    :param qmin:
    :param qbin:
    :return:
    """
    from ase.atoms import Atoms as atoms
    n = len(atoms)
    qmax_bin = int(qmax / qbin)
    symbols = atoms.get_chemical_symbols()
    e_num = atoms.get_atomic_numbers()

    scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)
    get_scatter_array(scatter_array, e_num, n, 0, qmax_bin, qbin)
    atoms.set_array('scatter', scatter_array)



def wrap_fq(atoms, qmax=25., qmin=0.0, qbin=.1):
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
    symbols = atoms.get_chemical_symbols()

    # define scatter_q information and initialize constants
    qmin_bin = int(qmin / qbin)
    qmax_bin = int(qmax / qbin) - qmin_bin
    scatter_q = np.arange(qmin, qmax, qbin)
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q, n)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d, n)

    # get scatter array
    scatter_array =atoms.get_array('scatter')

    # get non-normalized fq
    fq = np.zeros(len(scatter_q))
    get_fq_array(fq, r, scatter_array, n, qmin_bin, qmax_bin, qbin)


    #Normalize fq
    norm_array = np.zeros((n, n, len(scatter_q)))
    get_normalization_array(norm_array, scatter_array, qmin_bin, qmax_bin, n)
    norm_array = norm_array.sum(axis=(0, 1))
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / (n * norm_array) * fq)
    np.seterr(**old_settings)
    fq[:qmin_bin] = 0.0
    return fq


def wrap_pdf(atoms, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the atomic pair distribution function
    
    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------
    
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    fq = wrap_fq(atoms, qmax, qmin, qbin)
    pdf0 = get_pdf_at_qmin(fq, rstep, qbin, np.arange(0, rmax, rstep), 0.0,
                           rmax)
    return pdf0, fq


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------
    
    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    g_calc, fq = wrap_pdf(atoms, qmax, qmin, qbin, rmax, rstep)
    rw, scale = get_rw(gobs, g_calc, weight=None)
    return rw, scale, g_calc, fq


def wrap_fq_grad(atoms, qmax=25., qmin=0.0, qbin=.1):
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
    symbols = atoms.get_chemical_symbols()

    # define scatter_q information and initialize constants
    qmin_bin = int(qmin / qbin)
    qmax_bin = int(qmax / qbin) - qmin_bin
    scatter_q = np.arange(qmin, qmax, qbin)
    n = len(q)

    # Get pair coordinate distance array
    d = np.zeros((n, n, 3))
    get_d_array(d, q, n)

    # Get pair distance array
    r = np.zeros((n, n))
    get_r_array(r, d, n)

    # get scatter array
    scatter_array =atoms.get_array('scatter')

    # get non-normalized FQ

    #Normalize FQ
    norm_array = np.zeros((n, n, len(scatter_q)))
    get_normalization_array(norm_array, scatter_array, qmin_bin, qmax_bin, n)
    norm_array = norm_array.sum(axis=(0, 1))
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

    dfq_dq = np.zeros((n, 3, len(scatter_q)))
    fq_grad_position(dfq_dq, d, r, scatter_array, norm_array, qmin_bin,
                     qmax_bin, qbin)
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            dfq_dq[tx, tz, :qmin_bin] = 0.0
            dfq_dq[tx, tz] = np.nan_to_num(
                1 / (n * norm_array) * dfq_dq[tx, tz])
    np.seterr(**old_settings)
    return dfq_dq


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01,
                 rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------
    
    grad_rw: float
        The gradient of the Rw value with respect to the atomic positions, 
        in percent
    
    """
    if rw is None:
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin, rmax,
                                       rstep)
    fq_grad = wrap_fq_grad(atoms, qmax, qmin, qbin)
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep), qmin,
             rmax)
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    # print 'scale:', scale
    return grad_rw