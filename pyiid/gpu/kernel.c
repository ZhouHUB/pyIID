#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
__device__ void generate_D(float* D, float* q){
    //This kernel generates each of the elements of the D matrix.
    //The D matrix holds all of the inter-atomic distances for each of
    //directions x, y, and z.  The matrix is anti-symmetric and the diagonal
    //consists of zeros, as an atoms distance with itself is zero.  This
    //matrix is produced and saved so it can be used later in the gradient
    //method.  Thus we only need to replace one row/column at a time instead
    //of recalculating D every time we change an atomic position.  The D
    //matrix is NxNx3
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    D[tx, ty, tz] = q[ty, tz] - q[tx, tz];
    }

__device__ void reduce_D(double* reduced_D, double* D){
    //This kernel reduces the D matrix from NxNx3 to NxN by converting the
    //coordinate wise distances to a total distance via x**2+y**2+z**2 =
    //d**2.  The resulting array should have zero diagonals and be symmetric.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    reduced_D[tx, ty] = sqrt(D[tx, ty, 0]*D[tx, ty, 0]+D[tx, ty, 1]*D[tx, ty, 1]
    +D[tx, ty, 2]*D[tx, ty, 2]);
}

__global__ get_rw(double RW, double* Gr, double* exp_data){
    //This function generates the RW value, a measure of the model fit to the
    //experimental data, which is used as the potential energy U for the
    //refinement.

    //Generate the top array
    rw_top<>(double* top, double* Gr, double* exp_data);
    //Generate the bottom array
    rw_bottom<>(double* bot, double* Gr);
    //reduce the Nx1 top array to a 1x1 array by summing all the elements of the array
    one_to_zero_sum<>(double red_top, double* top);
    //reduce the Nx1 bottom array to a 1x1 array by summing all the elements 
    //of the array
    one_to_zero_sum<>(double red_bot, double* bot);
    RW = sqrt(red_top/red_bot);
}

__device__ rw_top(double top, double* Gr, double* exp_data){
    int tx = threadIdx.x;
    //Get the distance between the experimental and calculated curves
    top[tx] = (exp_data[tx]- Gr[tx])*(exp_data[tx]- Gr[tx]);
}

__device__ rw_bottom(double bot, double* Gr, double* exp_data){
    //Get the absolute distance of the experimental curve
    int tx = threadIdx.x
    bot[tx] = (exp_data[tx])*(exp_data[tx])
}

__device__ get_FQ_matrix(double* FQ, double* reduced_D, double*
scatter_array, double* Q, int Q_min_bin_number){
    //Generate F(Q) for each atom pair and Q value which will be summed.
    //reduced_D holds the absolute distances between the atoms.
    //scatter_array is a QxN array which holds the values of the scatter
    //functions for each of the atoms, where Q is the number of Q bins.
    //This could be made more efficient since usually the number of
    //constitiuent elements is smaller than the total number of atoms but
    //currently we don't keep track of which position goes with which
    //elemental signature, making the reduction of QxN to QxE difficult.

    //Note: F(Q) where Q<Qmin is zero
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    //NEED TO PAD F WITH ZEROS FROM 0 TO QMIN
    if (tx == ty or tz <= Q_min_bin_number){
        FQ3[tx, ty, tz] = 0
        }
    else{
        //technically scatter_array[ty] is the complex conjugate,
        //see Simon's book/papers
        FQ3[tx, ty, tz] = scatter_array[tx, tz]* scatter_array[ty, tz]*
        Sin(Q[tz]*reduced_D[tx, ty])/reduced_D[tx, ty]
        }
}

__global__ get_F(double* F, double* reduced_D, double* scatter_matrix,
double ave_scatter, double* Q, double qmin, double qstep){
    //This gets the NxNxQ FQ3 matrix, which describes the F(Q) contributuion
    //of each atom, at each Q vector and reduces it to a Qx1 F(Q) array,
    //which is ready for FFT to the G(r).

    //Need round up division
    int Q_min_bin_number = ROUND_UP_DIV(qmin/qstep)
    get_FQ_matrix<>(FQ3, reduced_D, scatter_matrix, Q, Q_min_bin_number)
    _syncthreads()
    //Reduce the FQ3 from NxNxQ to NxQ by summing along rows
    three_to_two_sum<>(FQ2, FQ3)
    _syncthreads()
    //Reduce the FQ2 from NxQ to 1xQ by summing along columns
    two_to_one_sum<>(F, FQ2)
    //Get F to be a Qx1 array
    transpose(F)
    _syncthreads()
    //Scattering prefactor
    prefactor = 1/len(reduced_D)/ave_scatter
    F *= prefactor
}
