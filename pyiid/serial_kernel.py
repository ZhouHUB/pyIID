__author__ = 'christopher'
import math
from numbapro import autojit
import mkl
from numbapro import cuda
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

try:
    cuda.select_device(0)
    targ = 'gpu'
except:
    targ = 'cpu'

@autojit(target=targ)
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

@autojit(target=targ)
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
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)

@autojit(target=targ)
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

@autojit(target=targ)
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
                    fq[kq] += smscale * \
                              dwscale * \
                              scatter_array[tx, kq] * \
                              scatter_array[ty, kq] / \
                              r[tx, ty] * \
                              math.sin(kq * Qbin * r[tx, ty])


@autojit(target=targ)
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
                norm_array[kq] += (
                    scatter_array[tx, kq] * scatter_array[ty, kq])
    norm_array *= 1. / (scatter_array.shape[0] ** 2)


# @autojit(target=targ)
def get_pdf_at_Qmin(fpad, rstep, Qstep, rgrid, qmin, rmax):
    # Zero all F values below qmin, which I think we have already done
    # nqmin = Qmin_bin
    # if nqmin > fpad.shape:
    #     nqmin = fpad.shape
    nfromdr = int(math.ceil(math.pi / rstep / Qstep))
    if nfromdr > int(len(fpad)):
        #put in a bunch of zeros
        fpad2 = np.zeros(nfromdr)
        fpad2[:len(fpad)] = fpad
        fpad = fpad2
    gpad = fftftog(fpad, Qstep, qmin)
    print len(gpad)
    drpad = math.pi/(len(gpad)*Qstep)
    pdf0 = np.zeros(len(rgrid), dtype=complex)
    for i, r in enumerate(rgrid):
        xdrp = r/drpad
        iplo = int(xdrp)
        iphi = iplo + 1
        wphi = xdrp - iplo
        wplo = 1.0 - wphi
    #     print 'i=', i
    #     print 'wphi=', wphi
    #     print 'wplo=', wplo
    #     print 'iplo=',iplo
    #     print 'iphi=', iphi
    #     print 'r=', r
        pdf0[i] = wplo * gpad[iplo] + wphi * gpad[iphi]
    #     print pdf0[i]
    pdf0 = np.abs(pdf0)
    pdf0 = pdf0[:len(pdf0)/2]*2
    return pdf0
    # return gpad

def fftftog(f, qstep, qmin):
    g = fftgtof(f, qstep, qmin)
    g *= 2.0/math.pi
    return g

def fftgtof(g, rstep, rmin):
    if g is None:
        return g
    padrmin = int(round(rmin / rstep))
    Npad1 = padrmin + len(g)
    # pad to the next power of 2 for fast Fourier transformation
    Npad2 = (1 << int(math.ceil(math.log(Npad1, 2))))
    print Npad2
    # // sine transformations needs an odd extension
    # // gpadc array has to be doubled for complex coefficients
    Npad4 = 4 * Npad2
    gpadc = np.zeros(Npad4)
    # // copy the original g signal
    ilo = padrmin
    for i in range(len(g)):
        gpadc[2*ilo] = g[i]
        ilo += 1
    # // copy the odd part of g skipping the first point,
    # // because it is periodic image of gpadc[0]
    ihi = 2 * Npad2 - 1
    ilo = 1
    for ilo in range(1, Npad2):
    # while ilo < Npad2:
        gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
        # ilo += 1
        ihi -= 1
    gpadcfft = np.fft.ihfft(gpadc,
                            # 2 * Npad2
    )
    # print gpadcfft
    # plt.plot(gpadcfft)
    # plt.show()
    f = np.zeros(Npad2, dtype=complex)
    for i in range(Npad2):
        f[i] = gpadcfft[2 * i + 1] * Npad2 * rstep
    return np.abs(f)

def get_dw_sigma_squared(s, u, r, d, n_range):
    for tx in n_range:
            for ty in n_range:
                rnormx = d[tx, ty, 0]/r[tx, ty]
                rnormy = d[tx, ty, 1]/r[tx, ty]
                rnormz = d[tx, ty, 2]/r[tx, ty]
                ux, uy, uz = u[tx] - u[ty]
                u_dot_r = rnormx * ux + rnormy * uy + rnormz * uz
                s[tx, ty] = u_dot_r * u_dot_r


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