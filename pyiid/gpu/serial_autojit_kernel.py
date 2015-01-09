__author__ = 'christopher'
import math
from numbapro import autojit
import mkl
from numbapro import cuda

try:
    cuda.select_device(0)
    targ = 'gpu'
except:
    targ = 'cpu'

@autojit(target=targ)
def d_kernel(a, b):
    return a - b

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
                d[tx, ty, tz] = d_kernel(q[ty, tz], q[tx, tz])



@autojit(target=targ)
def r_kernel(d):
    return math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)

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
            r[tx, ty] = r_kernel(d[tx, ty, :])


@autojit(target=targ)
def scatter_kernel(symbol, kq, Qbin, dpc):
     return dpc.scatteringfactortable.lookup(
                symbol, q=kq * Qbin)
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
            scatter_array[tx, kq] = scatter_kernel(symbols[tx], kq, Qbin, dpc)



@autojit(target=targ)
def fq_kernel(smscale, dwscale, sc1, sc2, r, kq, Qbin):
    return smscale * dwscale * sc1 * sc2 / r * math.sin(kq * Qbin * r)

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
                    # dwscale = math.exp(-.5 * dw_signal_sqrd * (kq*Qbin)**2)
                    fq[kq] += fq_kernel(smscale, dwscale, scatter_array[tx, kq], scatter_array[ty, kq], r[tx, ty], kq, Qbin)

@autojit(target=targ)
def norm_kernel(sa1, sa2):
    return sa1 * sa2
def get_normalization_array(norm_array, scatter_array, Qmax_Qmin_bin_range, n_range):
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
                norm_array[kq] += norm_kernel(scatter_array[tx, kq], scatter_array[ty, kq])
    norm_array *= 1. / (scatter_array.shape[0] ** 2)

@autojit(target=targ)
def get_pdf_at_Qmin(fpad):
    # Zero all F values below qmin, which I think we have already done
    # nqmin = qmin_bin
    # if nqmin > fpad.shape:
    #     nqmin = fpad.shape
    nfromdr = int(ceil(pi / rstep / Qstep))
    if nfromdr > int(fpad.size()):
        #put in a bunch of zeros
        pass

    gpad = fftftog(fpad, Qstep)

def fftftog(f, qstep, qmin):
    g = fftgtof(f, qstep, qmin)
    g *= 2.0/pi
    return g

def fftgtof(g, rstep, rmin):
    padrmin = int(round(rmin/rstep))
    Npad1 = padrmin + g.size()
    Npad2 =

@autojit(target=targ)
def get_dw_sigma_squared(s, u, r, d, n_range):
    for tx in n_range:
            for ty in n_range:
                rnormx = d[tx, ty, 0]/r[tx, ty]
                rnormy = d[tx, ty, 1]/r[tx, ty]
                rnormz = d[tx, ty, 2]/r[tx, ty]
                ux, uy, uz = u[tx] - u[ty]
                u_dot_r = rnormx * ux + rnormy * uy + rnormz * uz
                s[tx, ty] = u_dot_r * u_dot_r

@autojit(target=targ)
def get_gr(gr, r, rbin, n_range):
    """
    Generate gr the histogram of the atomic distances

    Parameters
    ----------
    gr: Nd array
    r: NxN array
    rbin: float
    n_range: Nd array
    :return:
    """
    for tx in n_range:
        for ty in n_range:
            gr[int(r[tx, ty] / rbin)] += 1