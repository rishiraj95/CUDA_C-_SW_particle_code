#include "cuda_kernels.h"

__device__ double2 fw_compute(){
	double2 fw = make_double2(0.0,0.0);
	return fw;

}

__global__ void EulerStep ( double* eta_j, 	const double* eta_k,   const double* eta_kx, 	const double* eta_ky,
							double* u_j, 	const double* u_k,     const double* u_kx, 		const double* u_ky,
							double* v_j, 	const double* v_k, 	   const double* v_kx, 		const double* v_ky,
						    double* t_j, 	const double* t_k, 	   const double* t_kx, 		const double* t_ky,
						    const double* H, const double* H_x, const double* H_y,
							const int Nx, 	const int Ny, 	const double dtk,		const double f, 	const double g) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int max_idx = Nx * Ny;
	if (idx < max_idx) {
		eta_j[idx] 	= eta_k[idx] 	- dtk * ((H[idx] + eta_k[idx])*(u_kx[idx]+ v_ky[idx]) + u_k[idx]*(H_x[idx]+eta_kx[idx]) + v_k[idx]*(H_y[idx]+eta_ky[idx]));
	    u_j[idx] 	= u_k[idx] 		- dtk * (-f*v_k[idx] + u_k[idx]*u_kx[idx] + v_k[idx]*u_ky[idx] + g*eta_kx[idx]);
	    v_j[idx] 	= v_k[idx] 		- dtk * (+f*u_k[idx] + u_k[idx]*v_kx[idx] + v_k[idx]*v_ky[idx] + g*eta_ky[idx]);
	    t_j[idx] 	= t_k[idx] 		- dtk * (u_k[idx]*t_kx[idx] + v_k[idx]*t_ky[idx] + t_k[idx]*u_kx[idx] + t_k[idx]*v_ky[idx]);
	}
}

__global__ void RK2( 	const double* eta_j, 	const double* eta_jx,   const double* eta_jy,   double* eta_k, 	const double* eta_kx,   const double* eta_ky,
					  	const double* u_j, 	    const double* u_jx,     const double* u_jy, 	double* u_k, 	const double* u_kx,     const double* u_ky,
					 	const double* v_j, 	    const double* v_jx, 	const double* v_jy, 	double* v_k, 	const double* v_kx, 	const double* v_ky,
						const double* t_j, 	    const double* t_jx, 	const double* t_jy, 	double* t_k, 	const double* t_kx, 	const double* t_ky,
						const double* H,  const double* H_x, const double* H_y,
						const int Nx, const int Ny, const double dt,  const double f, const double g) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int max_idx = Nx * Ny;

	double2 fw_j;
	double2 fw_k;

	fw_j=fw_compute();
	fw_k=fw_compute();

	double u_k_tmp;
	double v_k_tmp;

	u_k_tmp=u_k[idx];
	v_k_tmp=v_k[idx];

	if (idx < max_idx) {
		eta_k[idx] = eta_k[idx] - 0.5*dt*((H[idx] + eta_k[idx])*(u_kx[idx]+ v_ky[idx]) + u_k_tmp*(H_x[idx]+eta_kx[idx]) + v_k_tmp*(H_y[idx]+eta_ky[idx])
											+ (H[idx] + eta_j[idx])*(u_jx[idx] + v_jy[idx]) + u_j[idx]*(H_x[idx]+eta_jx[idx]) + v_j[idx]*(H_y[idx]+eta_jy[idx]));
		u_k[idx] 	= u_k_tmp 	- 0.5*dt*(-f*v_k_tmp + u_k_tmp*u_kx[idx] + v_k_tmp*u_ky[idx] + g*eta_kx[idx] - fw_k.x
											+ -f*v_j[idx] + u_j[idx]*u_jx[idx] + v_j[idx]*u_jy[idx] + g*eta_jx[idx] - fw_j.x);
		v_k[idx]	= v_k_tmp 	- 0.5*dt*(+f*u_k_tmp + u_k_tmp*v_kx[idx] + v_k_tmp*v_ky[idx] + g*eta_ky[idx] - fw_k.y
											+ +f*u_j[idx] + u_j[idx]*v_jx[idx] + v_j[idx]*v_jy[idx] + g*eta_jy[idx] - fw_j.y);
		t_k[idx]	= t_k[idx]	- 0.5*dt*(u_k_tmp*t_kx[idx] + v_k_tmp*t_ky[idx] + t_k[idx]*u_kx[idx] + t_k[idx]*v_ky[idx]
											+ u_j[idx]*t_jx[idx] + v_j[idx]*t_jy[idx] + t_j[idx]*u_jx[idx] + t_j[idx]*v_jy[idx]);
	}
}

__global__ void CalcDerivs(const cuDoubleComplex* VAR, cuDoubleComplex* VARX, cuDoubleComplex* VARY, const double* k, const double* l, const int Nx, const int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = Ny*(Nx/2+1);
	int jj = idx/(Nx/2+1);
	int ii = idx%(Nx/2+1);
	if (idx < max_idx) {
	    VARX[idx].x = -k[ii] * cuCimag(VAR[idx])/(Nx*Ny);
	    VARX[idx].y =  k[ii] * cuCreal(VAR[idx])/(Nx*Ny);
	    VARY[idx].x = -l[jj] * cuCimag(VAR[idx])/(Nx*Ny);
	    VARY[idx].y =  l[jj] * cuCreal(VAR[idx])/(Nx*Ny);
	}
}

__global__ void takeFilter(cuDoubleComplex* VAR, const double* filter, const int Nx, const int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = Ny*(Nx/2+1);
	if (idx < max_idx) {
	    VAR[idx].x = filter[idx] * cuCreal(VAR[idx])/(Nx*Ny);
	    VAR[idx].y = filter[idx] * cuCimag(VAR[idx])/(Nx*Ny);
	}
}

// this function has H as argument but does nto actually use it
__global__ void BuildTi(double* eta_k, double* u_k, double* v_k, double* t_k, const double* d_x_rand, const double* d_y_rand, const double* d_w_rand, const double* d_v_rand, const int Nx, const int Ny, const double dx, const double dy, const double L, const double H, const double U) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int max_idx = Nx*Ny;
	int jj = idx/Nx;
	int ii = idx%Nx;
	if (idx < max_idx) {
		eta_k[idx] 	= e_0(ii, jj, Nx, Ny, dx, dy, L, U);
		u_k[idx] 	= u_0(ii, jj, Nx, Ny, dx, dy, L, U);
        v_k[idx] 	= v_0(ii, jj, Nx, Ny, dx, dy, L, U);
        t_k[idx] 	= tracer_0(ii, jj, Nx, Ny, dx, dy, L, U);
    }
}

__global__ void ParticleTi(double* part_pos_x, double* part_pos_y, const int num_particles, const double Lx, const double Ly) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int max_idx = num_particles;
	const double Lx_spacing = Lx/(sqrt((double) num_particles)+1);
	const double Ly_spacing = Ly/(sqrt((double) num_particles)+1);
	if (idx < max_idx) {
		part_pos_x[idx] = Lx_spacing*(1+floor(idx/sqrt((double) num_particles)));
		part_pos_y[idx] = Ly_spacing*(1+fmod((double) idx,sqrt((double) num_particles)));
    }
}

__host__ __device__ double e_0 (const int ii, const int jj, const int Nx, const int Ny, const double dx, const double dy,  const double L, const double U) {
    // Initial height configuration
    double x = dx*(ii) - dx*Nx/2;
    double y = dy*(jj)- dy*Ny/2;
    double W = dy*Ny/10;
    double r = sqrt(x*x +y*y);
    double a0 = 0.1*0.5; // 0.1*H
    double wr = 0.4*0.5*sqrt(dx*Nx*dx*Nx + dy*Ny*dy*Ny);
    return a0*1/(pow(cosh(y/W),2)) + a0*pow(1/cosh(r/wr),8);
    // return 1.0*9.81*0.01*1e-4*L*U*( exp(-(x*x+y*y)/pow(L,2)));
    // return 0.;
}

__host__ __device__ double u_0 (const int ii, const int jj, const int Nx, const int Ny, const double dx, const double dy,  const double L, const double U) {
    // Initial horizontal velocity configuration
     // double x = dx*(ii) - dx*Nx/2;
     double y = dy*(jj) - dy*Ny/2;
     double W = dy*Ny/10;
     double a0 = 0.1*0.5; // 0.1*H
     double g = 9.81;
     double f = 1.0;
     // return 1.0*U*exp(-(y*y)/25^2);
     // return U/L*y*2*exp(-(x*x+y*y)/pow(L,2));
     return 2*g/f*a0/W*1/(pow(cosh(y/W),2))*tanh(y/W);
	// return 0.0;
}

__host__ __device__ double v_0(const int ii, const int jj, const int Nx, const int Ny, const double dx, const double dy,  const double L, const double U) {
    // Initial vertical velocity configuration
    double x = dx*(ii) - dx*Nx/2;
    double y = dy*(jj) - dy*Ny/2;
    // return 1.0*U*exp(-(x*x)/(L*L));
    // return -U/L*x*2*exp(-(x*x+y*y)/pow(L,2));
    return 0.0*x + 0.0*y;
}

__host__ __device__ double tracer_0 (const int ii, const int jj, const int Nx, const int Ny, const double dx, const double dy,  const double L, const double U) {
    // Initial tracer configuration
    // double x = dx*(ii)/(L/2);
    // double y = dy*(jj)/(L/2);
    // return 1* exp(-(x*x+y*y));
    // return 1.;
    return 0.;
}

__global__ void particle_interp_evolve(const double* u, const double* v, double* part_pos_x, double* part_pos_y, double* interp_u, double* interp_v, const int Nx, const int Ny, const int num_particles, const double dt, const double dx, const double dy) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = num_particles;
    if (idx < max_idx) {
    	int bl_x_index = floor((part_pos_x[idx])/dx);
    	int bl_y_index = floor((part_pos_y[idx])/dy);
    	interp_u[idx] =   u[bl_x_index+bl_y_index*Nx]*(1-(part_pos_x[idx]-(bl_x_index*dx))/dx)*(1-(part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ u[(bl_x_index+1)+(bl_y_index+1)*Nx]*((part_pos_x[idx]-(bl_x_index*dx))/dx)*((part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ u[(bl_x_index+1)+bl_y_index*Nx]*(part_pos_x[idx]-(bl_x_index*dx))/dx*(1-(part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ u[bl_x_index+(bl_y_index+1)*Nx]*(1-(part_pos_x[idx]-(bl_x_index*dx))/dx)*((part_pos_y[idx]-(bl_y_index*dy))/dy);
		interp_v[idx] =   v[bl_x_index+bl_y_index*Nx]*(1-(part_pos_x[idx]-(bl_x_index*dx))/dx)*(1-(part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ v[(bl_x_index+1)+(bl_y_index+1)*Nx]*((part_pos_x[idx]-(bl_x_index*dx))/dx)*((part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ v[(bl_x_index+1)+bl_y_index*Nx]*(part_pos_x[idx]-(bl_x_index*dx))/dx*(1-(part_pos_y[idx]-(bl_y_index*dy))/dy)
    					+ v[bl_x_index+(bl_y_index+1)*Nx]*(1-(part_pos_x[idx]-(bl_x_index*dx))/dx)*((part_pos_y[idx]-(bl_y_index*dy))/dy);
	    part_pos_x[idx] += dt*interp_u[idx];
	    part_pos_y[idx] += dt*interp_v[idx];
	    part_pos_x[idx] = fmod(part_pos_x[idx],dx*Nx);
	    part_pos_y[idx] = fmod(part_pos_y[idx],dy*Ny);
    }
}

__global__ void particle_interp_evolve_better(const double* u, const double* v, double* part_pos_x, double* part_pos_y, double* interp_u, double* interp_v, const int Nx, const int Ny, const int num_particles, const double dt, const double dx, const double dy) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int max_idx = num_particles;
    if (idx < max_idx) {
    	int x_index_0 = floor((part_pos_x[idx])/dx);
    	int y_index_0 = floor((part_pos_y[idx])/dy);
    	int x_index_m1 = (x_index_0 - 1)%Nx;
    	int y_index_m1 = (y_index_0 - 1)%Ny;
    	int x_index_1 = (x_index_0 + 1)%Nx;
    	int y_index_1 = (y_index_0 + 1)%Ny;
    	int x_index_2 = (x_index_1 + 1)%Nx;
    	int y_index_2 = (y_index_1 + 1)%Ny;
    	interp_u[idx] = cubic_interp_1d(cubic_interp_1d(u[x_index_m1+y_index_m1*Nx], u[x_index_0+y_index_m1*Nx], u[x_index_1+y_index_m1*Nx], u[x_index_2+y_index_m1*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(u[x_index_m1+y_index_0*Nx], u[x_index_0+y_index_0*Nx], u[x_index_1+y_index_0*Nx], u[x_index_2+y_index_0*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(u[x_index_m1+y_index_1*Nx], u[x_index_0+y_index_1*Nx], u[x_index_1+y_index_1*Nx], u[x_index_2+y_index_1*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(u[x_index_m1+y_index_2*Nx], u[x_index_0+y_index_2*Nx], u[x_index_1+y_index_2*Nx], u[x_index_2+y_index_2*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									(part_pos_y[idx]-y_index_0*dy)/dy);
    	part_pos_x[idx] += dt*interp_u[idx];
    	part_pos_x[idx] = fmod(fmod(part_pos_x[idx],dx*Nx) + dx*Nx,dx*Nx);
    	x_index_0 = floor((part_pos_x[idx])/dx);
    	x_index_m1 = (x_index_0 - 1)%Nx;
    	x_index_1 = (x_index_0 + 1)%Nx;
    	x_index_2 = (x_index_1 + 1)%Nx;
    	interp_v[idx] = cubic_interp_1d(cubic_interp_1d(v[x_index_m1+y_index_m1*Nx], v[x_index_0+y_index_m1*Nx], v[x_index_1+y_index_m1*Nx], v[x_index_2+y_index_m1*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(v[x_index_m1+y_index_0*Nx], v[x_index_0+y_index_0*Nx], v[x_index_1+y_index_0*Nx], v[x_index_2+y_index_0*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(v[x_index_m1+y_index_1*Nx], v[x_index_0+y_index_1*Nx], v[x_index_1+y_index_1*Nx], v[x_index_2+y_index_1*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									cubic_interp_1d(v[x_index_m1+y_index_2*Nx], v[x_index_0+y_index_2*Nx], v[x_index_1+y_index_2*Nx], v[x_index_2+y_index_2*Nx], (part_pos_x[idx] - x_index_0*dx)/dx),
    									(part_pos_y[idx]-y_index_0*dy)/dy);
	    part_pos_y[idx] += dt*interp_v[idx];
	    part_pos_y[idx] = fmod(fmod(part_pos_y[idx],dy*Ny) + dy*Ny,dy*Ny);
    }
}

__host__ __device__ double cubic_interp_1d(const double f_m1, const double f_0, const double f_1, const double f_2, const double interp_pt) {
	double a = -0.5*f_m1 + 1.5*f_0 - 1.5*f_1 + 0.5*f_2;
	double b = f_m1 - 2.5*f_0 + 2*f_1 - 0.5*f_2;
	double c = -0.5*f_m1 + 0.5*f_1;
	double d = f_0;
	return a*pow(interp_pt,3) + b*pow(interp_pt,2) + c*interp_pt + d;
}

__global__ void interactions_grid(double *part_pos_x, double *part_pos_y, int *grid_dim, int *bucket_begin, int *bucket_end, double *box_size, int *atom_index, int *num_interactions, int *start_interactions, double r ,int num_particles,int flagstore, int *storepair1, int *storepair2) {
//This functions checks for particles within a threshold distance and stores them as interacting pairs in lists.

    int i,j;
    int suminteractions;
    int startpoint;
    double dx,dy,dist;

    int gx,gy,xshift,yshift,gx_shift,gy_shift,jbin;
    i = blockIdx.x * blockDim.x + threadIdx.x;

//Convert the spatial position of particles into grid index of the buckets
    gx=int( part_pos_x[i]/ (box_size[0]/grid_dim[0]) );
    gy=int( part_pos_y[i]/ (box_size[1]/grid_dim[1]) );

//We do the computation for each particle in  parallel. The global unique thread index hence has to be less than the number of particles.
    if (i<num_particles){

//startpoint for each particle is needed to specify the particle indices in the storing lists.
        if (flagstore==1) startpoint=start_interactions[i];

        suminteractions=0;

// --------------------------------------

//Loops for checking adjacent buckets to a bucket in x and y direction.
for(xshift=-1;xshift<2;xshift++){
for(yshift=-1;yshift<2;yshift++){

//Convert particle positions in adjacent buckets to bucket grid index.
    gx_shift =( ( (gx+xshift) % grid_dim[0] ) + grid_dim[0] ) % grid_dim[0];
    gy_shift =( ( (gy+yshift) % grid_dim[1] ) + grid_dim[1] ) % grid_dim[1];
//Create 1-D index for the buckets    
    jbin=gx_shift+grid_dim[0]*gy_shift;

//bucket_begin[k] contains the index of the first particle in bucket k;  
//analogy follows for bucket_end.
//Check distances among particles in neighbouring buckets
        for (j=bucket_begin[jbin];j<bucket_end[jbin];j++){
            dx=part_pos_x[i]-part_pos_x[j];
            
//Take care of periodic boundary conditions            
            if(dx>box_size[0]/2.0) dx=box_size[0]-dx;
            if(dx<-box_size[0]/2.0) dx=box_size[0]+dx;
            dy=part_pos_y[i]-part_pos_y[j];
//Take care of periodic boundary conditions            
            if(dy>box_size[1]/2.0) dy=box_size[1]-dy;
            if(dy<-box_size[1]/2.0) dy=box_size[1]+dy;

            dist=sqrt(dx*dx+dy*dy);


            if (dist < r && i!=j){

                if (flagstore==1){
                    storepair1[startpoint+suminteractions]=atom_index[i];
                    storepair2[startpoint+suminteractions]=atom_index[j];
                }
                suminteractions = suminteractions + 1;
            }
        }
// ----------------------------------
}}

        if (flagstore==0) num_interactions[i]=suminteractions;

    } // i<num_particles
}


__global__ void interactions_n2(double *part_pos_x, double *part_pos_y, double *box_size, int *atom_index, int *num_interactions, int *start_interactions, double r ,int num_particles,int flagstore, int *storepair1, int *storepair2) {
    int i,j;
    int suminteractions;
    int startpoint;
    double dx,dy,dist;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<num_particles){

        if (flagstore==1) startpoint=start_interactions[i];

        suminteractions=0;
        for (j=0;j<num_particles;j++){
            dx=part_pos_x[i]-part_pos_x[j];
            if(dx>box_size[0]/2.0) dx=box_size[0]-dx;
            if(dx<-box_size[0]/2.0) dx=box_size[0]+dx;
            dy=part_pos_y[i]-part_pos_y[j];
            if(dy>box_size[1]/2.0) dy=box_size[1]-dy;
            if(dy<-box_size[1]/2.0) dy=box_size[1]+dy;

            dist=sqrt(dx*dx+dy*dy);

// can apply periodic shift metric here

            if (dist < r && i!=j){

                if (flagstore==1){
                    storepair1[startpoint+suminteractions]=atom_index[i];
                    storepair2[startpoint+suminteractions]=atom_index[j];
                }
                suminteractions = suminteractions + 1;
            }
        }

        if (flagstore==0) num_interactions[i]=suminteractions;

    } // i<num_particles
}
