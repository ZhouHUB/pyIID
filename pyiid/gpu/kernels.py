__author__ = 'christopher'
"""This file is to demonstrate the CUDA kernels written in python, so that
the can be turned into viable cuda code, without my ignorance of the C
langauge preventing me from expressing my thought process"""


def Gr_grad(grad, q, exp_data, delta_qi):
    get_Gr_U(U, q, exp_data)
    i = 0
    N = len(q)
    for i in N:
        for j in 3:
            #race condition problem here
            new_q = q
            new_q[i] += delta_qi
            #I am not certain where everything should be going here in terms
            # of keeping track of the variable memory
            #Also get_Gr_U in this place is rather wasteful, need to move to
            # more efficient method, only recalcuating parts of the distance
            # matrix that changed
            get_Gr_U(U, new_q, exp_data)
            #This won't work, this would require me to wait for each get_Gr_U
            #  to finish before starting the next batch
            grad[i] = (new_U-U)/delta_qi


def get_Gr_U(U, q, exp_data, scatter_matrix, ave_scatter):
    generate_D(D, q);
    reduce_D(reduced_D, D);
    get_Gr(Gr, reduced_D, scatter_matrix, ave_scatter);
    get_rw(RW, Gr, exp_data);


def get_rw(RW, Gr, exp_data):
    rw_top(top, Gr, exp_data)
    rw_bottom(bot, Gr)
    one_to_zero_sum(red_top, top)
    one_to_zero_sum(red_bot, bot)
    RW = (red_top/red_bot)**2
    
def rw_top(top, Gr, exp_data):
    int tx = threadIdx.x
    top[tx] = (exp_data[tx]- Gr[tx])**2
    
def rw_bottom(bot, Gr, exp_data):
    int tx = threadIdx.x
    bot[tx] = (exp_data[tx]- Gr[tx])**2

#generate D
def generate_D(D, q):
    int tx = threadIdx.x
    int ty = threadIdx.y
    int tz = threadIdx.z

    D[tx, ty, tz] = q[ty, tz] - q[tx, tz]

def reduce_D(reduced_D, D):
    int tx = threadIdx.x
    int ty = trheadIdx.y
    reduced_D[tx, ty] = sqrt(D[tx, ty, 0]**2+D[tx, ty, 1]**2+D[tx, ty, 2]**2)



def get_gr(gr, reduced_D, binsize, rmax):
    """Something about CUDA histogram creation"""
    for i in len(reduced_D):
        for j in range(i+len(reduced_D)):
            hist[reduced_D[i,j]] += 2
    return hist

def get_Gr(Gr, reduced_D,scatter_matrix, ave_scatter):
    N = len(reduced_D)
    get_F(F, reduced_D, scatter_matrix, qmin, qmax)
    Gr = cuda.sinfft(F)
    Gr *= 1/pi #may be nessisary

def get_F(F, reduced_D, scatter_matrix, ave_scatter, Q):
    get_FQ_matrix(FQ3, reduced_D, scatter_matrix, Q)
    _syncthreads()
    three_to_two_sum(FQ2, FQ3)
    _syncthreads()
    two_to_one_sum(F, FQ2)
    _syncthreads()
    prefactor = 1/len(reduced_D)/ave_scatter
    F *= prefactor
    "NEED TO PAD F WITH ZEROS FROM 0 TO QMIN"


def get_FQ_matrix(FQ, reduced_D, scatter_array, Q):
    int tx = threadIdx.x
    int ty = threadIdx.y
    int tz = threadIdx.z
    if tx == ty:
        FQ3[tx, ty, tz] = 0
    else:
        #technicly scatter_array[ty] is the complex conjugate
        FQ3[tx, ty, tz] = scatter_array[tx]*scatter_array[ty]*Sin(Q[
                                                                      tz]*reduced_D[tx,
                                                                ty])/reduced_D[tx,
                                                                          ty]

'''__device__ void generate_D(double* D, double* q){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    D[tx, ty, tz] = q[ty, tz] - q[tx, tz];
}'''

#generate reduced D
'''__device__ void reduce_D(double* reduced_D, double* D){
    int tx = threadIdx.x;
    int ty = trheadIdx.y;
    reduced_D[tx, ty] = sqrt(D[tx, ty, 0]**2+D[tx, ty, 1]**2+D[tx, ty, 2]**2);
}'''

'''__global__ get_rw(double RW, double* Gr, double* exp_data){
    rw_top<>(double* top, double* Gr, double* exp_data);
    rw_bottom<>(double* bot, double* Gr);
    one_to_zero_sum<>(double red_top, double* top);
    one_to_zero_sum<>(double red_bot, double* bot);
    RW = (red_top/red_bot)**2;
}'''
'''
__device__ rw_top(double top, double* Gr, double* exp_data){
    int tx = threadIdx.x;
    top[tx] = (exp_data[tx]- Gr[tx])**2;
}'''
'''
__device__ rw_bottom(double bot, double* Gr, double* exp_data){
    int tx = threadIdx.x
    bot[tx] = (exp_data[tx]- Gr[tx])**2
}'''
'''
__device__ get_FQ_matrix(double* FQ, double* reduced_D, double*
scatter_array, double* Q){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    if (tx == ty){
        FQ3[tx, ty, tz] = 0
        }
    else{
        #technicly scatter_array[ty] is the complex conjugate
        FQ3[tx, ty, tz] = scatter_array[tx]*scatter_array[ty]*Sin(Q[
                                                                      tz]*reduced_D[tx,
                                                                ty])/reduced_D[tx,
                                                                          ty]
        }
}'''
'''
__global__ get_F(double* F, double* reduced_D, double* scatter_matrix,
double ave_scatter, double* Q){
    get_FQ_matrix<>(FQ3, reduced_D, scatter_matrix, Q)
    _syncthreads()
    three_to_two_sum<>(FQ2, FQ3)
    _syncthreads()
    two_to_one_sum<>(F, FQ2)
    _syncthreads()
    prefactor = 1/len(reduced_D)/ave_scatter
    F *= prefactor
    //NEED TO PAD F WITH ZEROS FROM 0 TO QMIN
}