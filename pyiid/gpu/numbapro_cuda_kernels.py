__author__ = 'christopher'
import math
from numbapro import autojit
# import mkl
from numbapro import cuda
import numpy as np
from scipy import interpolate
import numpy.linalg as lg
from numba import *

targ = 'cpu'
cuda.select_device(1)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_d_array(d, q):
    """
    Generate the NxNx3 array which holds the coordinate pair distances

    Parameters
    ----------
    r: NxNx3 array
    q: Nx3 array
        The atomic positions
    n: 1d array
        Range of atomic numbers
    """
    tx, ty = cuda.grid(2)
    n = len(d)
    if tx >= n or ty >= n:
        return
    for tz in range(3):
        d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


@cuda.jit(argtypes=[f4[:, :], f4[:, :, :]])
def get_r_array(r, d):
    """
    Generate the Nx3 array which holds the pair distances

    Parameters
    ----------
    r: Nx3 array
    d: NxNx3 array
        The coordinate pair distances
    n: 1d array
        Range of atomic numbers
    """
    tx, ty = cuda.grid(2)
    n = len(d)
    if tx >= n or ty >= n:
        return
    r[tx, ty] = math.sqrt(
        d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty, 2] ** 2)


@autojit(target='cpu')
def get_scatter_array(scatter_array, symbols, dpc, n, qmin_bin, qmax_bin,
                      qbin):
    """
    Generate the scattering array, which holds all the Q dependant scatter
    factors

    Parameters:
    ---------
    scatter_array: NxM array
        Holds the scatter factors
    symbols: Nd array
        Holds the string reps of the atomic symbols
    dpc: DebyePDFCalculator instance
        The gives method to get atomic scatter factors
    n: int
        Number of atoms
    qmin_bin: int
        Binned scatter vector minimum
    qmax_bin: int
        Binned scatter vector maximum
    qbin: float
        The qbin size
    """
    for tx in range(n):
        for kq in range(qmin_bin, qmax_bin):
            scatter_array[tx, kq] = dpc.scatteringfactortable.lookup(
                symbols[tx], q=kq * qbin)


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :], f4])
def get_fq_p0(fq, r, qbin):
    """
    Get part of the reduced structure factor.  The FQ calculation is broken up because of GPU register issues
    :param fq:
    :param r:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(0, qmax_bin):
        fq[tx, ty, kq] = kq * qbin * r[tx, ty]

@cuda.jit(argtypes=[f4[:, :, :]])
def get_fq_p1(fq):
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
    n: Nd array
        The range of number of atoms
    qmin_bin: int
        Binned scatter vector minimum
    qmax_bin: int
        Binned scatter vector maximum
    qbin: float
        The qbin size
    """

    tx, ty = cuda.grid(2)
    n = len(fq)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(qmax_bin):
        fq[tx, ty, kq] = math.sin(fq[tx, ty, kq])


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_fq_p2(fq, r):
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
    n: Nd array
        The range of number of atoms
    qmin_bin: int
        Binned scatter vector minimum
    qmax_bin: int
        Binned scatter vector maximum
    qbin: float
        The qbin size
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(0, qmax_bin):
        fq[tx, ty, kq] /= r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :]])
def get_fq_p3(fq, norm_array):
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
    n: Nd array
        The range of number of atoms
    qmin_bin: int
        Binned scatter vector minimum
    qmax_bin: int
        Binned scatter vector maximum
    qbin: float
        The qbin size
    """

    tx, ty = cuda.grid(2)

    n = len(norm_array)
    qmax_bin = fq.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return

    for kq in range(qmax_bin):
        fq[tx, ty, kq] *= norm_array[tx, ty, kq]

# '''
@cuda.jit(argtypes=[f4[:, :, :], f4[:, :]])
def get_normalization_array(norm_array, scat):
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
     n: Nd array
        The range of number of atoms
    """

    tx, ty = cuda.grid(2)
    n = len(scat)

    if tx >= n or ty >= n:
        return
    qmax_bin = scat.shape[1]
    for kq in range(qmax_bin):
        norm_array[tx, ty, kq] = scat[tx, kq] * scat[ty, kq]
# '''

def get_pdf_at_qmin(fpad, rstep, qstep, rgrid, qmin, rmax):
    """
    Get the atomic pair distribution function

    Parameters
    -----------
    :param fpad: 1d array
        The reduced structure function, padded with zeros to qmin
    :param rstep: float
        The step size in real space
    :param qstep: float
        The step size in inverse space
    :param rgrid: 1d array
        The real space r values
    :param qmin: float
        The minimum q value
    :param rmax: float
        The maximum r vlaue
    Returns
    -------
    1d array:
        The atomic pair distributuion function
    """
    # Zero all F values below qmin, which I think we have already done
    # nqmin = qmin_bin
    # if nqmin > fpad.shape:
    # nqmin = fpad.shape
    nfromdr = int(math.ceil(math.pi / rstep / qstep))
    if nfromdr > int(len(fpad)):
        # put in a bunch of zeros
        fpad2 = np.zeros(nfromdr)
        fpad2[:len(fpad)] = fpad
        fpad = fpad2
    gpad = fft_fq_to_gr(fpad, qstep, qmin)
    # # print len(gpad)
    drpad = math.pi / (len(gpad) * qstep)
    pdf0 = np.zeros(len(rgrid), dtype=complex)
    for i, r in enumerate(rgrid):
        xdrp = r / drpad / 2
        iplo = int(xdrp)
        iphi = iplo + 1
        wphi = xdrp - iplo
        wplo = 1.0 - wphi
        # # print 'i=', i
        # # print 'wphi=', wphi
        # # print 'wplo=', wplo
        # # print 'iplo=',iplo
        # # print 'iphi=', iphi
        # # print 'r=', r
        pdf0[i] = wplo * gpad[iplo] + wphi * gpad[iphi]
    # # print pdf0[i]
    pdf1 = pdf0 * 2
    return pdf1
    # return gpad


def fft_fq_to_gr(f, qstep, qmin):
    g = fft_gr_to_fq(f, qstep, qmin)
    g *= 2.0 / math.pi
    return g


def fft_gr_to_fq(g, rstep, rmin):
    if g is None:
        return g
    pad_rmin = int(round(rmin / rstep))
    npad1 = pad_rmin + len(g)
    # pad to the next power of 2 for fast Fourier transformation
    npad2 = (1 << int(math.ceil(math.log(npad1, 2))))
    # print npad2
    # // sine transformations needs an odd extension
    # // gpadc array has to be doubled for complex coefficients
    npad4 = 4 * npad2
    gpadc = np.zeros(npad4)
    # // copy the original g signal
    ilo = pad_rmin
    for i in range(len(g)):
        gpadc[2 * ilo] = g[i]
        ilo += 1
    # // copy the odd part of g skipping the first point,
    # // because it is periodic image of gpadc[0]
    ihi = 2 * npad2 - 1
    ilo = 1
    for ilo in range(1, npad2):
        # while ilo < npad2:
        gpadc[2 * ihi] = -1 * gpadc[2 * ilo]
        # ilo += 1
        ihi -= 1
    gpadcfft = np.fft.ihfft(gpadc,
                            # 2 * npad2
    )
    # # print gpadcfft
    # plt.plot(gpadcfft)
    # plt.show()
    f = np.zeros(npad2, dtype=complex)
    for i in range(npad2):
        f[i] = gpadcfft[2 * i + 1] * npad2 * rstep
    return f.imag


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

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :, :], f4[:,:]])
def fq_grad_position0(rgrad, d, r):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        rgrad[tx, ty, tz] = d[tx, ty, tz] / r[tx, ty]

@cuda.jit(argtypes=[f4[:, :, :], f4])
def fq_grad_position1(q_over_r, qbin):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param q_over_r: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(q_over_r)
    qmax_bin = q_over_r.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        q_over_r[tx, ty, kq] = kq * qbin


@cuda.jit(argtypes=[f4[:, :, :], f4[:,:]])
def fq_grad_position2(q_over_r, r):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param q_over_r: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = q_over_r.shape[3]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        q_over_r[tx, ty, kq] /= r[tx, ty]



@cuda.jit(argtypes=[f4[:, :, :], f4[:, :,:]])
def fq_grad_position3(cos_term, kqr):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] = math.cos(kqr[tx, ty, kq])

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :,:]])
def fq_grad_position4(cos_term, q_over_r):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] *= q_over_r[tx, ty, kq]

@cuda.jit(argtypes=[f4[:, :, :], f4[:, :,:]])
def fq_grad_position5(cos_term, norm):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] *= norm[tx, ty, kq]

@cuda.jit(argtypes=[f4[:, :, :], f4[:,:]])
def fq_grad_position6(fq, r):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(r)
    qmax_bin = r.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        fq[tx, ty, kq] /= r[tx, ty]


@cuda.jit(argtypes=[f4[:, :, :], f4[:,:, :]])
def fq_grad_position7(cos_term, fq):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(cos_term)
    qmax_bin = cos_term.shape[2]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for kq in range(qmax_bin):
        cos_term[tx, ty, kq] -= fq[tx, ty, kq]



@cuda.jit(argtypes=[f4[:, :, :, :], f4[:,:, :]])
def fq_grad_position_final1(grad_p, rgrad):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(grad_p)
    qmax_bin = grad_p.shape[3]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        for kq in range(qmax_bin):
            grad_p[tx, ty, tz, kq] = rgrad[tx, ty, tz]

@cuda.jit(argtypes=[f4[:, :, :, :], f4[:,:, :]])
def fq_grad_position_final2(grad_p, cos_term):
    """
    Generate the gradient F(Q) for an atomic configuration
    :param grad_p: Nx3xQ numpy array
        The array which will store the FQ gradient
    :param d:
        The distance array for the configuration
    :param r:
    :param scatter_array:
    :param norm_array:
    :param qmin_bin:
    :param qmax_bin:
    :param qbin:
    :return:
    """

    tx, ty = cuda.grid(2)
    n = len(grad_p)
    qmax_bin = grad_p.shape[3]
    if tx >= n or ty >= n:
        return
    if tx == ty:
        return
    for tz in range(3):
        for kq in range(qmax_bin):
            grad_p[tx, ty, tz, kq] *= cos_term[tx, ty, kq]


def grad_pdf(grad_pdf, grad_fq, rstep, qstep, rgrid, qmin, rmax):
    n = len(grad_fq)
    for tx in range(n):
        for tz in range(3):
            grad_pdf[tx, tz] = get_pdf_at_qmin(grad_fq[tx, tz], rstep, qstep,
                                               rgrid, qmin, rmax)


# @autojit(target=targ)
def get_rw(gobs, gcalc, weight=None):
    # print np.dot(gcalc.T, gcalc)
    scale = np.dot(np.dot(1. / (np.dot(gcalc.T, gcalc)), gcalc.T), gobs)
    if weight is None:
        weight = np.ones(gcalc.shape)
    top = np.sum(weight[:] * (gobs[:] - scale * gcalc[:]) ** 2)
    bottom = np.sum(weight[:] * gobs[:] ** 2)
    return np.sqrt(top / bottom).real, scale


def get_grad_rw(grad_rw, grad_pdf, gcalc, gobs, rw, scale, weight=None):
    if weight is None:
        weight = np.ones(gcalc.shape)
    n = len(grad_pdf)
    for tx in range(n):
        for tz in range(3):
            # part1 = 1.0/np.sum(weight[:]*(scale*gcalc[:]-gobs[:])**2)
            part1 = 1.0 / np.sum(weight[:] * (scale * gcalc[:] - gobs[:]))
            # print('part1', part1)
            # part2 = np.sum(scale*(scale*gcalc[:] - gobs[:])*grad_pdf[tx, tz, :])
            part2 = np.sum(scale * grad_pdf[tx, tz, :])
            # print('part2', part2)
            grad_rw[tx, tz] = rw.real / part1.real * part2.real


if __name__ == '__main__':
    from ase.atoms import Atoms as atoms
    import ase.io as aseio
    from ase.visualize import view
    import matplotlib.pyplot as plt
    from timeit import default_timer as time

    atoms = aseio.read(
        '/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/C60.xyz')
    q = atoms.get_positions()
    qmin = .25
    qmax = 25
    qbin = .1

    q = q.astype(np.float32)
    n = len(q)
    qmin_bin = 0
    qmax_bin = int(qmax / qbin)

    # Atoms definition, outside of calc

    # scatter_array = np.zeros((n, qmax_bin))
    # scatter_array = np.ones((n, qmax_bin), dtype=np.float32) * 2
    scatter_array = np.loadtxt('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/1-C60.d/c60_scat.txt', dtype=np.float32)
    # get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
    atoms.set_array('scatter', scatter_array)

    d = np.zeros((n, n, 3), dtype=np.float32)
    r = np.zeros((n, n), dtype=np.float32)
    norm_array = np.zeros((n, n, qmax_bin), dtype=np.float32)
    tzr = np.zeros((n, n, qmax_bin), dtype=np.float32)
    print tzr.shape[2]

    # get_scatter_array(scatter_array, atoms.get_chemical_symbols(), dpc, n, qmax_bin, qbin)
    atoms.set_array('scatter', scatter_array)

    super_fq = np.zeros((n, n, qmax_bin), dtype=np.float32)
    print super_fq.shape

    stream = cuda.stream()
    tpb = 32
    bpg = int(math.ceil(float(n) / tpb))
    print(qmax_bin, len(r), bpg * tpb)

    s = time()
    #push empty d, full q, and number n to GPU
    dscat = cuda.to_device(scatter_array, stream)
    dnorm = cuda.to_device(norm_array)

    get_normalization_array[(bpg, bpg), (tpb, tpb), stream](dnorm, dscat)
    cuda.synchronize()
    dnorm.to_host(stream)
    print norm_array