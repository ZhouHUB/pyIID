__author__ = 'christopher'
"""
The basic idea for this module is a place to store GPU accelerated code using
NumbaPro and CUDA.
"""
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")


def pair_distance(w1, w2):
    diff = w1-w2




def gpu_get_pdf_sub_matrix(q_sub_coord):
    """
    Generate the PDF Matrix which contains all the atomic pair distances

    Parameters
    ----------
    q: Nx3 ndarray
        The positions of the atoms in 3D, N is the number of atoms

    Returns
    -------
    ndarray:
        NxNx3 dimensional array each NxN slice contains all the pairs in a
        coordinate e.g. one slice = [[x0-0, x0-1, x0-2,...], [x1-0, x1-1,...]]
        Note that the matrix is symmetric with respect to the diagonal and
        all diagonal elements are 0
    """
    N = q.shape[0]
    startX, startY, startZ = cuda.grid(3)
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;

    blksz = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    gridsz = int(math.ceil(float(n) / blksz))
    stream = cuda.stream()
    d_M = cuda.device_array(n, dtype=np.double, stream=stream)


def get_pdf_matrix(q):
    #The list of x, y, z matrix for the PDF
    M_list = []
    #for x, y, and z
    for i in range(3):
        #symmetric matrix which holds pair distance in x, y, or z
        M_w = np.zeros((len(q), len(q)))
        #for all atoms
        for j in len(q):
            #for all atoms greater than first atom
            for k in range(j+1, len(q)):
                if j ==k:
                    M_w[j, k] = 0
                else:
                    M_w[j, k] = q[j]-q[k]
                    M_w[k, j] = M_w[j, k]
        M_list.append(M_w)
    return M_list


def dist_from_pdf_matrix(M_list):
    D = np.zeros((len(M_list[0],len(M_list[0]))))
    X, Y, Z = M_list
    for j in len(X):
        for k in range(j+len(X)):
            D[j, k] = np.sqrt(X[j, k]**2 + Y[j, k]**2 + Z[j, k]**2)
            D[k, j] = D[j, k]
    return D


def get_PDF(D, binsize, rmax):
    hist = np.zeros((rmax/binsize))
    for i in len(D):
        for j in range(i+len(D)):
            hist[D[i,j]] += 2
    return hist


def rw(pdf_calc, pdf_exp, weight=None):
    if weight is None:
        weight = np.ones(pdf_calc.shape)
    top = 0
    bottom = 0
    for i in len(pdf_calc):
        top += weight[i]*(pdf_exp[i]-pdf_calc[i])**2
        bottom += weight[i]*pdf_exp[i]**2
    return np.sqrt(top/bottom)


def grad_pdf(q, D, exp_data, dq, binsize, rmax):
    grad = np.zeros((len(q), 3))
    #for atoms
    for i in len(q):
        #for all dimensions x,y,z
        for j in range(3):
        #move single atom in x, y, or z
            q[i, j] += dq
            D_qi = D
            #calculate that atom's pairs and put them into atom_pairs
            atom_pairs = np.zeros(len(q))
            for k in len(q):
                if i == k:
                    dist = 0
                else:
                    dist = np.sqrt(np.dot(q[i]-q[k]))
                #change the atom's distances in the D matrix
                D_qi[i, k] = dist
                D_qi[k, i] = dist
            #get PDF for q(x,y,or z)+dq(x,y,or z)
            hist = get_PDF(D_qi, binsize, rmax)
            RW = rw(hist, exp_data, weight=None)
            grad[i, j] = RW
