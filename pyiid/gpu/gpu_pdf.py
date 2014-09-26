__author__ = 'christopher'
import numpy as np


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
