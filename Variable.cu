#include "Variable.h"

Variable::Variable (const int Nx, const int Ny) {
    // Allocate on Host
    h_var  = (double*)malloc(Nx*Ny*sizeof(double));
    h_varx = (double*)malloc(Nx*Ny*sizeof(double));
    h_vary = (double*)malloc(Nx*Ny*sizeof(double));
    // Allocate on Device
    cudaMalloc(&d_var,Nx*Ny*sizeof(double));
    cudaMalloc(&d_varx,Nx*Ny*sizeof(double));
    cudaMalloc(&d_vary,Nx*Ny*sizeof(double));
    cudaMalloc(&VAR,(Nx/2+1)*Ny*sizeof(cuDoubleComplex));
    cudaMalloc(&VARX,(Nx/2+1)*Ny*sizeof(cuDoubleComplex));
    cudaMalloc(&VARY,(Nx/2+1)*Ny*sizeof(cuDoubleComplex));
    // CUFFT Plans
    cufftPlan2d(&fft_var, Ny, Nx, CUFFT_D2Z);
    cufftPlan2d(&ifft_VARX, Ny, Nx, CUFFT_Z2D);
    cufftPlan2d(&ifft_VARY, Ny, Nx, CUFFT_Z2D);
    cufftPlan2d(&ifft_VAR, Ny, Nx, CUFFT_Z2D);
};

Variable::~Variable () {
    cufftDestroy(fft_var);
    cufftDestroy(ifft_VARX);
    cufftDestroy(ifft_VARY);
    free(h_var);
    free(h_varx);
    free(h_vary);
    cudaFree(d_var);
    cudaFree(d_varx);
    cudaFree(d_vary);
    cudaFree(VAR);
    cudaFree(VARX);
    cudaFree(VARY);
};

double Variable::MaxVar (const int Nx, const int Ny) {
    // Calculate the maximum velocity for adaptive dt
    double max_var = fabs(h_var[0]);
    for (int ii=1; ii<Nx*Ny; ii++) {
        if (max(fabs(h_var[ii]),max_var) != max_var) {
            max_var = fabs(h_var[ii]);
        }
    }
    return max_var;
}

void Variable::ComputeDerivatives (const double* k, const double* l, const int Nx, const int Ny, const int tpb, const int nblks) {
    // Fourier transform
    cufftExecD2Z(fft_var, d_var, VAR);
    // Calculate derivatives
    CalcDerivs<<<nblks,tpb>>>(VAR, VARX, VARY, k, l, Nx, Ny);
    // Transform back
    cufftExecZ2D(ifft_VARX, VARX, d_varx);
    cufftExecZ2D(ifft_VARY, VARY, d_vary);
}

void Variable::Filter (const double* filter, const int Nx, const int Ny, const int tpb, const int nblks) {
    // Fourier transform
    cufftExecD2Z(fft_var, d_var, VAR);
    // Apply filter
    takeFilter<<<nblks,tpb>>>(VAR, filter, Nx, Ny); 
    // Transform back     
    cufftExecZ2D(ifft_VAR, VAR, d_var);
}