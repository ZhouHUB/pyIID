__author__ = 'christopher'
from pyiid.serial_kernel import *
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
dpc = DebyePDFCalculator()


def wrap_FQ(atoms, Qmax=25., Qmin=0.0, Qbin=.1, rmax=40., rstep=.01):
    q = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # define Q information and initialize constants
    Qmin_bin = int(Qmin / Qbin)
    Qmax_bin = int(Qmax / Qbin) - Qmin_bin
    Q = np.arange(Qmin, Qmax, Qbin)
    N = len(q)

    # Get pair coordinate distance array
    d = np.zeros((N, N, 3))
    get_d_array(d, q, N)

    #Get pair distance array
    r = np.zeros((N, N))
    get_r_array(r, d, N)

    #get scatter array
    scatter_array = np.zeros((N, len(Q)))
    get_scatter_array(scatter_array, symbols, dpc, N, Qmin_bin, Qmax_bin, Qbin)

    #get non-normalized FQ
    fq = np.zeros(len(Q))
    get_fq_array(fq, r, scatter_array, N, Qmin_bin, Qmax_bin, Qbin)

    #Normalize FQ
    norm_array = np.zeros(len(Q))
    get_normalization_array(norm_array, scatter_array, Qmin_bin, Qmax_bin, N)
    old_settings = np.seterr(all='ignore')
    FQ = np.nan_to_num(1 / (N * norm_array) * fq)
    np.seterr(**old_settings)
    return FQ

def wrap_pdf(atoms, Qmax=25., Qmin=0.0, Qbin=.1, rmax=40., rstep=.01):
    FQ = wrap_FQ(atoms, Qmax, Qmin, Qbin, rmax, rstep)
    pdf0 = get_pdf_at_Qmin(FQ, rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)
    return pdf0, FQ


def wrap_rw(atoms, gobs, Qmax=25., Qmin=0.0, Qbin=.1, rmax=40., rstep=.01):
    gcalc, FQ = wrap_pdf(atoms, Qmax, Qmin, Qbin, rmax, rstep)
    rw, scale = get_rw(gobs, gcalc, weight=None)
    return rw*100, scale, gcalc, FQ


def wrap_FQ_grad(atoms, Qmax=25., Qmin=0.0, Qbin=.1):
    q = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # define Q information and initialize constants
    Qmin_bin = int(Qmin / Qbin)
    Qmax_bin = int(Qmax / Qbin)
    Q = np.arange(Qmin, Qmax, Qbin)
    N = len(q)


    #initialize constants
    N = len(q)
    # Get pair coordinate distance array
    d = np.zeros((N, N, 3))
    get_d_array(d, q, N)

    #Get pair distance array
    r = np.zeros((N, N))
    get_r_array(r, d, N)

    #get scatter array
    scatter_array = np.zeros((N, len(Q)))
    get_scatter_array(scatter_array, symbols, dpc, N, Qmin_bin, Qmax_bin, Qbin)

    #get non-normalized FQ

    #Normalize FQ
    norm_array = np.zeros(len(Q))
    get_normalization_array(norm_array, scatter_array, Qmin_bin, Qmax_bin, N)

    grad_p = np.zeros((N, 3, len(Q)))
    fq_grad_position(grad_p, d, r, scatter_array, norm_array, Qmin_bin, Qmax_bin, Qbin)
    old_settings = np.seterr(all='ignore')
    for tx in range(N):
        for tz in range(3):
            grad_p[tx, tz] = np.nan_to_num(1 / (N * norm_array) * grad_p[tx, tz])
    np.seterr(**old_settings)
    return grad_p


def wrap_grad_rw(atoms, gobs, Qmax=25., Qmin=0.0, Qbin=.1, rmax=40., rstep=.01, rw=None, gcalc=None, scale=None):
    if rw is None:
        rw, scale, gcalc, FQ = wrap_rw(atoms, gobs, Qmax, Qmin, Qbin, rmax, rstep)
    fq_grad = wrap_FQ_grad(atoms, Qmax, Qmin, Qbin)
    pdf_grad = np.zeros((len(atoms), 3, rmax/rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    # print 'scale:', scale
    return grad_rw*100