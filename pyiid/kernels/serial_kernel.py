__author__ = 'christopher'
import math
from numbapro import autojit
from numba import *
import mkl
import numpy as np
import xraylib

targ = 'cpu'

# F(Q) test_kernels -----------------------------------------------------------

@autojit(target=targ)
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    d: NxNx3 array
    q: Nx3 array
        The atomic positions
    """
    n = len(q)
    for tx in range(n):
        for ty in range(n):
            for tz in range(3):
                d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


@autojit(target=targ)
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    """
    n = len(r)
    for tx in range(n):
        for ty in range(n):
            r[tx, ty] = math.sqrt(
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@autojit(target=targ)
def get_scatter_array(scatter_array, numbers, qbin):
    """
    Generate the scattering array, which holds all the Q dependant scatter
    factors
    
    Parameters:
    ---------
    scatter_array: NxQ array
        Holds the scatter factors
    numbers: Nd array
        Holds the atomic numbers
    qbin: float
        The qbin size
    """
    n = len(scatter_array)
    qmax_bin = scatter_array.shape[1]
    for tx in range(n):
        for kq in range(0, qmax_bin):
            # note xraylib uses q = sin(th/2)
            # as opposed to our q = 4pi sin(th/2)
            scatter_array[tx, kq] = xraylib.FF_Rayl(numbers[tx],
                                                    kq * qbin / 4 /np.pi)


@autojit(target=targ)
def get_fq_array(fq, r, scatter_array, qbin):
    """
    Generate F(Q), not normalized, via the Debye sum

    Parameters:
    ---------
    fq: Nd array
        The reduced scatter pattern
    r: NxN array
        The pair distance array
    scatter_array: NxM array
        The scatter factor array
    qbin: float
        The qbin size
    """
    sum_scale = 1
    n = len(r)
    qmax_bin = len(fq)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for kq in range(0, qmax_bin):
                    debye_waller_scale = 1
                    # TODO: debye_waller_scale = math.exp(
                    # -.5 * dw_signal_sqrd * (kq*Qbin)**2)
                    fq[kq] += sum_scale * \
                              debye_waller_scale * \
                              scatter_array[tx, kq] * \
                              scatter_array[ty, kq] / \
                              r[tx, ty] * \
                              math.sin(kq * qbin * r[tx, ty])


@autojit(target=targ)
def get_normalization_array(norm_array, scatter_array):
    """
    Generate the Q dependant normalization factors for the F(Q) array

    Parameters:
    -----------
    norm_array: NxNxQ array
        Normalization array
    scatter_array: NxQ array
        The scatter factor array
    """
    n = len(norm_array)
    qmax_bin = norm_array.shape[2]

    for kq in range(0, qmax_bin):
        for tx in range(n):
            for ty in range(n):
                norm_array[tx, ty, kq] = (
                    scatter_array[tx, kq] * scatter_array[ty, kq])

# PDF test_kernels ------------------------------------------------------------

@autojit(target=targ)
def get_pdf_at_qmin(fpad, rstep, qstep, rgrid):
    """
    Get the atomic pair distribution function

    Parameters
    -----------
    fpad: 1d array
        The reduced structure function, padded with zeros to qmin
    rstep: float
        The step size in real space
    qstep: float
        The step size in inverse space
    rgrid: 1d array
        The real space r values
    rmax: float
        The maximum r value

    Returns
    -------
    1d array:
        The atomic pair distribution function
    """
    nfromdr = int(math.ceil(math.pi / rstep / qstep))
    if nfromdr > int(len(fpad)):
        # put in a bunch of zeros
        fpad2 = np.zeros(nfromdr)
        fpad2[:len(fpad)] = fpad
        fpad = fpad2
    gpad = fft_fq_to_gr(fpad, qstep)
    drpad = math.pi / (len(gpad) * qstep)
    pdf0 = np.zeros(len(rgrid), dtype=complex)
    for i, r in enumerate(rgrid):
        xdrp = r / drpad / 2
        iplo = int(xdrp)
        iphi = iplo + 1
        wphi = xdrp - iplo
        wplo = 1.0 - wphi
        pdf0[i] = wplo * gpad[iplo] + wphi * gpad[iphi]
    pdf1 = pdf0 * 2
    return pdf1
    # return gpad


@autojit(target='cpu')
def fft_fq_to_gr(f, qbin):
    """
    Fourier Transform from F(Q) to G(r)

    Parameters
    -----------
    f: Nd array
        F(Q)
    qbin: float
        Qbin size
    qmin:

    Returns
    -------
    g: Nd array
        The PDF
    """
    g = fft_gr_to_fq(f, qbin)
    g *= 2.0 / math.pi
    return g


@autojit(target='cpu')
def fft_gr_to_fq(g, rbin):
    """
    Fourier Transform from G(r) to F(Q)

    Parameters
    ----------
    g: Nd array
        The PDF
    rbin: float
        The size of the distance bins

    Returns
    -------
    f: Nd array
        The reduced structure factor
    """
    if g is None:
        return g
    npad1 = len(g)
    # pad to the next power of 2 for fast Fourier transformation
    npad2 = (1 << int(math.ceil(math.log(npad1, 2))))
    # sine transformations needs an odd extension
    # gpadc array has to be doubled for complex coefficients
    npad4 = 4 * npad2
    gpadc = np.zeros(npad4)
    # copy the original g signal
    ilo = 0
    for i in range(len(g)):
        gpadc[2 * ilo] = g[i]
        ilo += 1
    # copy the odd part of g skipping the first point,
    # because it is periodic image of gpadc[0]
    ihi = 2 * npad2 - 1
    for ilo in range(1, npad2):
        gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
        ihi -= 1
    gpadcfft = np.fft.ihfft(gpadc)
    f = np.zeros(npad2, dtype=complex)
    for i in range(npad2):
        f[i] = gpadcfft[2 * i + 1] * npad2 * rbin
    return f.imag


def get_rw(gobs, gcalc, weight=None):
    """
    Get the rw value for the PDF

    Parameters
    -----------
    gobs: Nd array
        The observed PDF
    gcalc: Nd array
        The model PDF
    weight: Nd array, optional
        The weight for the PDF

    Returns
    -------
    float:
        The rw
    scale:
        The scale factor in the rw calculation
    """
    # Note: The scale is set to 1, having a variable scale seems to create
    # issues with the potential energy surface.
    # scale = np.dot(np.dot(1. / (np.dot(gcalc.T, gcalc)), gcalc.T), gobs)
    scale = 1
    if weight is None:
        weight = np.ones(gcalc.shape)
    top = np.sum(weight[:] * (gobs[:] - scale * gcalc[:]) ** 2)
    bottom = np.sum(weight[:] * gobs[:] ** 2)
    return np.sqrt(top / bottom).real, scale


def get_chi_sq(gobs, gcalc):
    """
    Get the rw value for the PDF

    Parameters
    -----------
    gobs: Nd array
        The observed PDF
    gcalc: Nd array
        The model PDF
    weight: Nd array, optional
        The weight for the PDF

    Returns
    -------
    float:
        The rw
    scale:
        The scale factor in the rw calculation
    """
    # Note: The scale is set to 1, having a variable scale seems to create
    # issues with the potential energy surface.
    # scale = np.dot(np.dot(1. / (np.dot(gcalc.T, gcalc)), gcalc.T), gobs)
    scale = 1
    return np.sum((gobs-gcalc)**2
                  # /gobs
    ).real, scale


# Gradient test_kernels -------------------------------------------------------
@autojit(target=targ)
def fq_grad_position(grad_p, d, r, scatter_array, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration

    Parameters
    ------------
    grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    d: NxNx3 array
        The distance array for the configuration
    r: NxN array
        The inter-atomic distances
    scatter_array: NxQ array
        The scatter factor array
    qbin: float
        The size of the Q bins
    """
    n = len(r)
    qmax_bin = grad_p.shape[2]
    for tx in range(n):
        for tz in range(3):
            for ty in range(n):
                if tx != ty:
                    for kq in range(0, qmax_bin):
                        sub_grad_p = \
                            scatter_array[tx, kq] * \
                            scatter_array[ty, kq] * \
                            d[tx, ty, tz] * \
                            (
                                (kq * qbin) *
                                r[tx, ty] *
                                math.cos(kq * qbin * r[tx, ty]) -
                                math.sin(kq * qbin * r[tx, ty])
                            ) \
                            / (r[tx, ty] ** 3)
                        grad_p[tx, tz, kq] += sub_grad_p


@autojit(target=targ)
def grad_pdf(grad_pdf, grad_fq, rstep, qstep, rgrid):
    n = len(grad_fq)
    for tx in range(n):
        for tz in range(3):
            grad_pdf[tx, tz] = get_pdf_at_qmin(grad_fq[tx, tz], rstep, qstep,
                                               rgrid)


def get_grad_rw(grad_rw, grad_pdf, gcalc, gobs, rw, scale, weight=None):
    """
    Get the gradient of the model PDF
    
    Parameters
    ------------
    grad_rw: Nx3 array
        Holds the gradient
    grad_pdf: Nx3xR array
        The gradient of the PDF
    gcalc: Nd array
        The calculated PDF
    gobs: Nd array
        The observed PDF
    rw: float 
        The current Rw value
    scale: float
        The current scale
    weight: nd array, optional
        The PDF weights
    """
    if weight is None:
        weight = np.ones(gcalc.shape)
    n = len(grad_pdf)
    for tx in range(n):
        for tz in range(3):
            '''
            # Wolfram alpha
            grad_rw[tx, tz] = np.sum((gcalc[:] - gobs[:]) * grad_pdf[tx, tz, :])/(np.sum(gobs[:]**2) * rw)
            grad_rw[tx, tz] = grad_rw[tx, tz].real
            '''
            # '''
            # Sympy
            grad_rw[tx, tz] = rw * np.sum((gcalc[:] - gobs[:]) * grad_pdf[tx, tz, :])/np.sum((gobs[:] - gcalc[:])**2)
            # '''
            '''
            # Previous version
            part1 = 1.0 / np.sum(weight[:] * (scale * gcalc[:] - gobs[:]))
            part2 = np.sum(scale * grad_pdf[tx, tz, :])
            grad_rw[tx, tz] = rw.real / part1.real * part2.real
            '''


def get_grad_chi_sq(grad_rw, grad_pdf, gcalc, gobs):
    """
    Get the gradient of the model PDF

    Parameters
    ------------
    grad_rw: Nx3 array
        Holds the gradient
    grad_pdf: Nx3xR array
        The gradient of the PDF
    gcalc: Nd array
        The calculated PDF
    gobs: Nd array
        The observed PDF
    rw: float
        The current Rw value
    scale: float
        The current scale
    weight: nd array, optional
        The PDF weights
    """
    n = len(grad_pdf)
    for tx in range(n):
        for tz in range(3):
            grad_rw[tx, tz] = np.sum(-2 * (gobs - gcalc) * grad_pdf[tx, tz, :]
                                     # /gobs
            )
# Misc. Kernels----------------------------------------------------------------

@autojit(target=targ)
def get_dw_sigma_squared(s, u, r, d, n):
    for tx in range(n):
        for ty in range(n):
            rnormx = d[tx, ty, 0] / r[tx, ty]
            rnormy = d[tx, ty, 1] / r[tx, ty]
            rnormz = d[tx, ty, 2] / r[tx, ty]
            ux = u[tx, 0] - u[ty, 0]
            uy = u[tx, 1] - u[ty, 1]
            uz = u[tx, 2] - u[ty, 2]
            u_dot_r = rnormx * ux + rnormy * uy + rnormz * uz
            s[tx, ty] = u_dot_r * u_dot_r


@autojit(target=targ)
def get_gr(gr, r, rbin, n):
    """
    Generate gr the histogram of the atomic distances

    Parameters
    ----------
    gr: Nd array
    r: NxN array
    rbin: float
    n: Nd array
    :return:
    """
    for tx in range(n):
        for ty in range(n):
            gr[int(r[tx, ty] / rbin)] += 1


def simple_grad(grad_p, d, r):
    """
    Gradient of the delta function gr
    grad_p:
    d:
    r:
    :return:
    """
    n = len(r)
    for tx in range(n):
        for ty in range(n):
            if tx != ty:
                for tz in range(3):
                    grad_p[tx, tz] += d[tx, ty, tz] / (r[tx, ty] ** 3)
