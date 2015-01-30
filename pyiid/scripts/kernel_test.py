import matplotlib.pyplot as plt
from pyiid.utils import convert_atoms_to_stru
import cProfile
# cProfile.run('''
__author__ = 'christopher'
import numpy as np
import ase.io as io
from diffpy.srreal.pdfcalculator import DebyePDFCalculator
from pyiid.serial_kernel import get_d_array, get_r_array, get_scatter_array, \
    get_fq_array, get_normalization_array, get_pdf_at_qmin, fq_grad_position
from scipy.fftpack import idst
from ase import Atoms

dpc = DebyePDFCalculator()

# Load Atoms
# atoms = io.read('/home/christopher/25_nm_half.xyz')
# atoms = io.read('/home/christopher/pdfgui_np_35.xyz')




# ''',
# sort='cumtime')
# dpc.rmax = 40

# plt.plot(np.arange(0, rmax, rstep), pdf0, rcalc, gcalc)
# plt.show()

# print FQ
# plt.plot(Q, FQ)
# plt.show()

def get_pdf(atoms):
    #extract data from atoms
    q = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # define Q information
    Qmax = 25.
    rmax = 40
    # Qmin = 2.5
    Qmin = 0.0
    Qbin = np.pi/(rmax+6*2*np.pi/Qmax)
    qmin_bin = int(Qmin / Qbin)
    Qmax_bin = int(Qmax / Qbin)
    Q = np.arange(Qmin, Qmax, Qbin)

    rstep = .01


    #initialize constants
    N = len(q)
    print N
    d = np.zeros((N, N, 3))
    n_range = range(N)
    range_3 = range(3)

    # Get pair coordinate distance array
    get_d_array(d, q, N)

    print 'd'
    print d

    #Get pair distance array
    r = np.zeros((N, N))
    get_r_array(r, d, N)
    print 'r'
    print r

    #get scatter array
    scatter_array = np.zeros((N, len(Q)))
    get_scatter_array(scatter_array, symbols, N, qmin_bin, Qmax_bin, Qbin)
    print 'sa'
    print scatter_array
    #remove self_scattering
    # np.fill_diagonal(scatter_array, 0)

    #get non-normalized FQ
    fq = np.zeros(len(Q))
    get_fq_array(fq, r, scatter_array, N, qmin_bin, Qmax_bin, Qbin)

    #Normalize FQ
    norm_array = np.zeros(len(Q))
    get_normalization_array(norm_array, scatter_array, qmin_bin, Qmax_bin, N)
    FQ = np.nan_to_num(1 / (N * norm_array) * fq)
    print(FQ)
    # pdf = idst(FQ)
    pdf0 = get_pdf_at_qmin(FQ, rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)
    rgrid = np.arange(0, rmax, rstep)
    return rgrid, pdf0
    # plt.plot(rgrid, pdf0)
    # plt.show()
    # rcalc, gcalc = dpc(convert_atoms_to_stru(atoms), qmin=Qmin)


    grad_p = np.zeros((N, 3, len(Q)))
    fq_grad_position(grad_p, d, r, scatter_array, norm_array, qmin_bin,
                     Qmax_bin, Qbin)
    print grad_p
    print grad_p.shape

    # pdf_grad_p = np.zeros((N, 3))
    # pdf_grad_position(pdf_grad_p, grad_p, rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)

    # plt.plot(grad_p[100, 0, :])
    # plt.show()
    # plt.imshow(grad_p[:,0,:])
    # plt.show()
    #
    # partial = get_pdf_at_Qmin(grad_p[0, 0, :], rstep, Qbin, np.arange(0, rmax, rstep), Qmin, rmax)
    # plt.plot(partial)
    # plt.show()

    # ''',
    # sort='cumtime')
    # dpc.rmax = 40

    # plt.plot(np.arange(0, rmax, rstep), pdf0, rcalc, gcalc)
    # plt.show()

    # print FQ
    # plt.plot(Q, FQ)
    # plt.show()
def get_FQ(atoms):
    q = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # define Q information
    Qmax = 25.
    rmax = 40
    # Qmin = 2.5
    Qmin = 0.0
    Qbin = np.pi/(rmax+6*2*np.pi/25)
    qmin_bin = int(Qmin / Qbin)
    Qmax_bin = int(Qmax / Qbin)
    Q = np.arange(Qmin, Qmax, Qbin)

    rstep = .01


    #initialize constants
    N = len(q)
    print N
    d = np.zeros((N, N, 3))
    n_range = range(N)
    range_3 = range(3)

    # Get pair coordinate distance array
    get_d_array(d, q, N)

    print 'd'
    print d

    #Get pair distance array
    r = np.zeros((N, N))
    get_r_array(r, d, N)
    print 'r'
    print r

    #get scatter array
    scatter_array = np.zeros((N, len(Q)))
    get_scatter_array(scatter_array, symbols, N, qmin_bin, Qmax_bin, Qbin)
    print 'sa'
    print scatter_array
    #remove self_scattering
    # np.fill_diagonal(scatter_array, 0)

    #get non-normalized FQ
    fq = np.zeros(len(Q))
    get_fq_array(fq, r, scatter_array, N, qmin_bin, Qmax_bin, Qbin)

    #Normalize FQ
    norm_array = np.zeros(len(Q))
    get_normalization_array(norm_array, scatter_array, qmin_bin, Qmax_bin, N)
    FQ = np.nan_to_num(1 / (N * norm_array) * fq)
    return Q, FQ

def get_FQ_dir(atoms):
    q = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # define Q information
    Qmax = 25.
    rmax = 40
    # Qmin = 2.5
    Qmin = 0.0
    Qbin = np.pi/(rmax+6*2*np.pi/25)
    qmin_bin = int(Qmin / Qbin)
    Qmax_bin = int(Qmax / Qbin)
    Q = np.arange(Qmin, Qmax, Qbin)

    rstep = .01


    #initialize constants
    N = len(q)
    print N
    d = np.zeros((N, N, 3))
    n_range = range(N)
    range_3 = range(3)

    # Get pair coordinate distance array
    get_d_array(d, q, N)

    print 'd'
    print d

    #Get pair distance array
    r = np.zeros((N, N))
    get_r_array(r, d, N)
    print 'r'
    print r

    #get scatter array
    scatter_array = np.zeros((N, len(Q)))
    get_scatter_array(scatter_array, symbols, N, qmin_bin, Qmax_bin, Qbin)
    print 'sa'
    print scatter_array
    #remove self_scattering
    # np.fill_diagonal(scatter_array, 0)

    #get non-normalized FQ
    fq = np.zeros(len(Q))
    get_fq_array(fq, r, scatter_array, N, qmin_bin, Qmax_bin, Qbin)

    #Normalize FQ
    norm_array = np.zeros(len(Q))
    get_normalization_array(norm_array, scatter_array, qmin_bin, Qmax_bin, N)
    FQ = np.nan_to_num(1 / (N * norm_array) * fq)
    grad_p = np.zeros((N, 3, len(Q)))
    fq_grad_position(grad_p, d, r, scatter_array, norm_array, qmin_bin,
                     Qmax_bin, Qbin)
    return grad_p

atoms = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
Q, FQ1 = get_FQ(atoms)
a = .00001
atoms2 = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0+a)])
Q, FQ2 = get_FQ(atoms2)
# plt.plot(FQ2-FQ1)
# plt.show()

# grad = get_FQ_dir(atoms)
# plt.plot(grad[2,1,:])
# plt.show()
# plt.plot(Q, (FQ2-FQ1)/a,
         # Q, grad[2,1,:]
# )
# plt.show()
# plt.plot(Q, FQ2-FQ1, Q, grad[2,2,:])
# plt.show()
atoms = Atoms('Au4', [(0,0,0), (3,0,0), (0,3,0), (3,3,0)])
Q, FQ1 = get_pdf(atoms)
a = .00001
plt.plot(Q, FQ1)
plt.show()
atoms2 = Atoms('Au4', [(0,0,0), (3,0,0), (0,3+a,0), (3,3,0)])
Q, FQ2 = get_pdf(atoms2)
plt.plot(Q, (FQ2-FQ1)/a,
         # Q, grad[2,1,:]
)
plt.show()