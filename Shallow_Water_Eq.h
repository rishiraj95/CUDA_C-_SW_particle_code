#ifndef SW_H
#define SW_H

#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <numeric> 
#include <cufftw.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include "Variable.h"
#include "Particle.h"
#include "cuda_kernels.h"

#include <netcdf.h>
#include <hdf5.h>

class Shallow_Water_Eq {
	public:
	    // Read-in parameters
		int Nx, Ny;
		int num_particles;
		double Hmax, g, f, parint;
		double Lx, Ly;
		double Ti, Tf, plot_interval;
	    // Time parameters
	    int timestep;
	    int cntr;
	    double tk;
	    double dtk;
	    double c0;
	    double next_output_time;
	    // CUDA parameters
	    int tpb;
	    int nblks;
	    // Grid parameters
	    double dx;
	    double dy;
	    double* xgrid;
	    double* ygrid;
	    // Wavenumber parameters
	    double dk; 
	    double k_nq;
	    double dl;
	    double l_nq;
	    double k_cut;
	    double l_cut;
	    double alpha;
	    double beta;
	    // Wavenumber and filter arrays
	    double* h_k; 
	    double* h_l;
	    double* h_filter; 
	    double* d_k; 
	    double* d_l;
	    double* d_filter; 
	    // Variables at two timesteps
	    Variable* e_j;
	    Variable* e_k;
	    Variable* u_j;
	    Variable* u_k;
	    Variable* v_j;
	    Variable* v_k;
	    Variable* tracer_j;
	    Variable* tracer_k;

            Variable* H_xy;


	    // Particles
	    double* trajectory;
	    Particle* particles;
	    // Energy Diagnostics
	    double total_ke;
	    double total_pe;
	    // Special array for vort
	    double* vort;
	    // Constructor and Destructor
	    Shallow_Water_Eq (int);
	    ~Shallow_Water_Eq ();
	    // Functions
	    void Print_To_File (const int);
            void writeInteractions (const int);
	    void doDerivatives ();
	    void doRK2();
	    void doParticle();
	    double adaptive_timestep();
	    void doEnergy();
	    void doVort();
	    void read_parameters(int*, int*, int*, double*, int*, double*, double*, double*, double*, double*, double*, double*, double*);
	    void read_variables(double**, double**, double**, double**, double**, double**, double**);
};
#endif
