__author__ = 'christopher'
import numpy as np
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from utils import convert_atoms_to_stru
import scipy.optimize as op


def scale_to_rw_min(scale, calc, exp):
    """
    Scale the PDF data so it is close to the calculated results, this is for
    the scalar minimization

    Parameters:
    -----------
    scale: float
        The scale factor
    calc: ndarray
        Calculated PDF
    exp: ndarray
        Experimental PDF
    Returns:
    --------
    float:
        The Rw value

    """
    return rw(calc, scale*exp)

def pdf_U(atoms, exp_data, bin_size=.01, rmax=40.):

    #get PDF from atoms
    hist = np.zeros((rmax/bin_size,))
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = atoms.get_distance(atoms[i], atoms[j])
            k = np.floor(dist/bin_size)
            hist[k] += 1
    #return chi_squared or Rw for exp vs. predicted


def Debye_srreal_U(atoms, exp_data, rmax):
    """
    Calculates the rw value using srreal for a set of atoms and experimental
    data

    Parameters:
    -----------
    atoms: ase.Atoms object
        The atomic configuration
    exp_data: ndarray
        The experimental PDF

    Returns:
    --------
    float:
        The Rw value
    """
    stru = convert_atoms_to_stru(atoms)
    dpc = DebyePDFCalculator()
    dpc.qmax = 25
    dpc.rmin = 0.00
    dpc.rmax = rmax
    r0, g0 = dpc(stru, qmin=2.5)
    #scale data to minimize rw
    res = op.minimize_scalar(scale_to_rw_min, bounds = (0, 5), args = (g0,
                                                                   exp_data),
                             method='Bounded')
    scale = res.x
    # exp_data *= np.max(g0)/np.max(exp_data)
    return res.fun


def one_atom_PDF(atom_index, atoms):
    for j in range(i+1, len(atoms)):
            dist = atoms.get_distance(atoms[i], atoms[j])
            k = np.floor(dist/bin_size)
            hist[k] += 1


def rw(pdf_calc, pdf_exp, weight=None):
    if weight is None:
        weight = np.ones(pdf_calc.shape)
    top = np.sum(weight[:]*(pdf_exp[:]-pdf_calc[:])**2)
    bottom = np.sum(weight[:]*pdf_exp[:]**2)
    return np.sqrt(top/bottom)

# def DebyePDF(atoms,qmin):
#     """S(Q)-1 = 1/N<f>**2*SUM(f*j fi*Sin(Q*rij)/Q/rij
#     F(Q) = Q[S(Q)-1]
#
#     """
#     rgrid = getRgrid()
#     pdf0 = getPDFAtQmin(qmin)
#     pdf1 = applyEnvelopes(rgrid, pdf0)
#     return pdf1
#
# def applyEnvelopes(x, y):
#     assert(x.size() == y.size())
#     z = y
#     EnvelopeStorage::const_iterator evit;
#     for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
#         PDFEnvelope& fenvelope = *(evit->second);
#         ::const_iterator xi = x.begin();
#         ::iterator zi = z.begin();
#         for (; xi != x.end(); ++xi, ++zi)
#             *zi *= fenvelope(*xi)
#     return z
# def getPDFAtQmin(qmin):
#     # // build a zero padded F vector that gives dr <= rstep
#      fpad = getF();
#     // zero all F values below qmin
#     int nqmin = pdfutils_qminSteps(qmin, getQstep());
#     if (nqmin > int(fpad.size()))  nqmin = fpad.size();
#     fill(fpad.begin(), fpad.begin() + nqmin, 0.0);
#     int nfromdr = int(ceil(M_PI / getRstep() / getQstep()));
#     if (nfromdr > int(fpad.size()))  fpad.resize(nfromdr, 0.0);
#      gpad = fftftog(fpad, getQstep());
#     const double drpad = M_PI / (gpad.size() * getQstep());
#      rgrid = getRgrid();
#      pdf0(rgrid.size());
#     ::const_iterator ri = rgrid.begin();
#     ::iterator pdfi = pdf0.begin();
#     for (; ri != rgrid.end(); ++ri, ++pdfi)
#     {
#         double xdrp = *ri / drpad;
#         int iplo = int(xdrp);
#         int iphi = iplo + 1;
#         double wphi = xdrp - iplo;
#         double wplo = 1.0 - wphi;
#         assert(iphi < int(gpad.size()));
#         *pdfi = wplo * gpad[iplo] + wphi * gpad[iphi];
#     }
#     return pdf0

def pdf_calc_U(atoms, exp_data, pdf_U_scale):
    """
    Calculate the total potential energy, combining PDF with a ab-initio
    structure calculator

    Parameters:
    ----------
    atoms: ase.Atoms object
        Atomic configuration
    exp_data: ndarray
        PDF data
    pdf_U_scale: float
        Scale to multiply the PDF U by in order to make pdf based potential
        comperable to the ab-initio potential

    Returns:
    -------
    float:
        The total energy
    """
    pdfu = pdf_U(atoms, exp_data)
    pdf_nrg = pdfu*pdf_U_scale
    calc_nrg = atoms.get_potential_energy()
    return pdf_nrg+calc_nrg
