#ifndef PART_H
#define PART_H

#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <cufftw.h>
#include <cuda.h>

#include "cuda_kernels.h"

class Particle {
	public:
		int num_particles;
		double* h_part_pos_x;
		double* h_part_pos_y;
		double* d_part_pos_x;
		double* d_part_pos_y;
		double* interp_u;
		double* interp_v;
		// Constructor and destructor
		Particle(const int);
		~Particle();
};


#endif
