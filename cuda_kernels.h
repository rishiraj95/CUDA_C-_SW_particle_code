#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cufft.h>
#include <cufftw.h>


__global__ void EulerStep(	double*, 	const double*, 	const double*, 	const double*,
							double*, 	const double*, 	const double*, 	const double*,
							double*, 	const double*, 	const double*, 	const double*,
						    double*, 	const double*, 	const double*, 	const double*,
						    const double*,    const double*,  const double*,
							const int, const int, const double, const double, const double);
__global__ void RK2(	const double*,	const double*, 	const double*, 	double*, 	const double*, 	const double*,
					 	const double*, 	const double*, 	const double*, 	double*, 	const double*, 	const double*,
					 	const double*, 	const double*, 	const double*, 	double*, 	const double*, 	const double*,
						const double*, 	const double*, 	const double*, 	double*, 	const double*, 	const double*,
                                                const double*,  const double*,  const double*,
						const int, const int, const double, const double, const double);
__global__ void CalcDerivs(const cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex*, const double*, const double*, const int, const int);
__global__ void takeFilter(cuDoubleComplex*, const double*, const int, const int);
__global__ void BuildTi(double*, double*, double*, double*, const double*, const double*, const double*, const double*, const int, const int, const double, const double, const double, const double, const double);
__global__ void ParticleTi(double*, double*, const int, const double, const double);
__host__ __device__ double  e_0 (const int, const int, const int, const int, const double, const double, const double, const double);
__host__ __device__ double u_0 (const int, const int, const int, const int, const double, const double, const double, const double);
__host__ __device__ double v_0 (const int, const int, const int, const int, const double, const double, const double, const double);
__host__ __device__ double tracer_0 (const int, const int, const int, const int, const double, const double, const double, const double);
__global__ void particle_interp_evolve(const double*, const double*, double*, double*, double*, double*, const int, const int, const int, const double, const double, const double);
__global__ void particle_interp_evolve_better(const double*, const double*, double*, double*, double*, double*, const int, const int, const int, const double, const double, const double);
__host__ __device__ double cubic_interp_1d(const double, const double, const double, const double, const double);
__global__ void interactions_grid(double*, double*, int*, int*, int*, double*, int*, int*, int*, double ,int ,int , int*, int*);
__global__ void interactions_n2(double*, double*, double*, int*, int*, int*, double ,int ,int , int*, int*);
#endif
