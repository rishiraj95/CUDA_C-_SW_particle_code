#include"Particle.h"

Particle::Particle(int in_num_particles) {
	// Pass number of particles to the Particle class
	num_particles = in_num_particles;
	// Allocate particle positions on the host
	h_part_pos_x = (double *)malloc(num_particles*sizeof(double));
    h_part_pos_y = (double *)malloc(num_particles*sizeof(double));
    // Allocate particle positions and interpolated velocities on device
    cudaMalloc(&d_part_pos_x,num_particles*sizeof(double));
    cudaMalloc(&d_part_pos_y,num_particles*sizeof(double));
    cudaMalloc(&interp_u,num_particles*sizeof(double));
    cudaMalloc(&interp_v,num_particles*sizeof(double));

}

Particle::~Particle() {
	// Free allocations
	cudaFree(d_part_pos_x);
	cudaFree(d_part_pos_y);
	cudaFree(interp_u);
	cudaFree(interp_v);
	free(h_part_pos_x);
	free(h_part_pos_y);
}