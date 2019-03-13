#ifndef VAR_H
#define VAR_H

#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cufftw.h>
#include <cuda.h>

#include "cuda_kernels.h"

class Variable {
    public:
		double* h_var;
        double* h_varx;
        double* h_vary;
        double* d_var;
        double* d_varx;
        double* d_vary;
        cufftDoubleComplex* VAR;
        cufftDoubleComplex* VARX;
        cufftDoubleComplex* VARY;
        cufftHandle fft_var;          
        cufftHandle ifft_VAR;          
        cufftHandle ifft_VARX;        
        cufftHandle ifft_VARY;
	    // Constructor and Destructor
	    Variable (const int, const int);
	    ~Variable ();
	    // Functions
        double MaxVar (const int, const int);
        void ComputeDerivatives (const double*, const double*, const int, const int, const int , const int);
        void Filter (const double*, const int, const int, const int, const int);
};
#endif