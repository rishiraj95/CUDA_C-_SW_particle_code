#include "Shallow_Water_Eq.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
// #define void NC_ERR(e) {fprintf(stderr,"Error: %s\n", nc_strerror(e));}

Shallow_Water_Eq::Shallow_Water_Eq (int k) {
    // Read parameters from NETCDF file
    read_parameters(&Nx, &Ny, &num_particles, &parint, &cntr, &Hmax, &g, &f, &Lx, &Ly, &Ti, &Tf, &plot_interval); 
    // Time parameters
    timestep = 0;
    tk = Ti;
    dtk = 0;
    c0 = sqrt(g*Hmax); 
    next_output_time = plot_interval + Ti;
    // CUDA parameters
    tpb = 512; // Threads per Block
    nblks = (Nx*Ny)/tpb + (((Nx*Ny)%tpb) ? 1 : 0); // Number of Blocks
    // Build grid and related parameters
    dx = Lx/Nx;
    dy = Ly/Ny;
    xgrid = new double[Nx];
    for (int ii=0; ii<Nx; ii++) {
        xgrid[ii] = dx*(ii);
    }
    ygrid = new double[Ny];
    for (int ii=0; ii<Ny; ii++) {
        ygrid[ii] = dy*(ii);
    }
    // Wavenumber and filter parameters
    dk = 2*M_PI/(Lx);
    k_nq = M_PI/dx;
    dl = 2*M_PI/(Ly);
    l_nq = M_PI/dy;
    k_cut = 0.45 * k_nq;
    l_cut = 0.45 * l_nq;
    alpha = 20.0;
    beta = 2.0;
    // Allocate wavenumber arrays
    // Since FFTW only uses (N/2+1) entries for one dimension we pick it to be the x direction
    // thus k is only size (Nx/2+1) 
    h_k = (double *)malloc(Nx*sizeof(double));
    h_l = (double *)malloc(Ny*sizeof(double));
    h_filter = (double *)malloc(Ny*(Nx/2+1)*sizeof(double));
    cudaMalloc(&d_k,Nx*sizeof(double));
    cudaMalloc(&d_l,Ny*sizeof(double));
    cudaMalloc(&d_filter,Ny*(Nx/2+1)*sizeof(double));
    // Build wavenumber vectors
    for (int ii=0; ii<Nx/2; ii++) {
      h_k[ii] = ii * dk;
      h_k[ii+(Nx/2)] = ( ii - (Nx/2) ) * dk;
    }
    h_k[Nx/2] = 0;
    for (int ii=0; ii<Ny/2; ii++) {
      h_l[ii] = ii * dl;
      h_l[ii+(Ny/2)] = ( ii - (Ny/2) ) * dl;
    }
    h_l[Ny/2] = 0;
    // Build filter
    double fx;
    double fy;
    for (int jj=0; jj<Ny; jj++) {
        if (jj == Ny/2) {
            fy = 0.0;
        }
        else if (fabs(h_l[jj]) < l_cut) {
            fy = 1.0;
        }
        else {
            fy = exp(-alpha*(pow((fabs(h_l[jj]) - l_cut)/(l_nq - l_cut), beta)));
        }
        for (int ii=0; ii<(Nx/2+1); ii++) {
            if (ii == Nx/2) {
                fx = 0.0;
            }
            else if (fabs(h_k[ii]) < k_cut) {
                fx = 1.0;
            }
            else {
                fx = exp(-alpha*(pow((fabs(h_k[ii]) - k_cut)/(k_nq - k_cut), beta)));
            }
            h_filter[ii+jj*(Nx/2+1)] = fx*fy;
        }
    }
    cudaMemcpy(d_k, h_k, Nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, Ny*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, Ny*(Nx/2+1)*sizeof(double), cudaMemcpyHostToDevice);
    // Initialize the primary variables for the two timesteps
    e_j = new Variable (Nx, Ny);
    e_k = new Variable (Nx, Ny);
    u_j = new Variable (Nx, Ny);
    u_k = new Variable (Nx, Ny);
    v_j = new Variable (Nx, Ny);
    v_k = new Variable (Nx, Ny);
    tracer_j = new Variable (Nx, Ny);
    tracer_k = new Variable (Nx, Ny);

    H_xy = new Variable (Nx, Ny);

    // Initialize the particles
    trajectory = new double[num_particles];
    for (int ii=0; ii<num_particles; ii++) {
        trajectory[ii] = ii;
    }
    particles = new Particle (num_particles);
    // Initialize the total energies to zero
    total_ke = 0.0;
    total_pe = 0.0;
    // Initialize the vort array
    vort = new double[Nx*Ny];
    // Read in the initial conditions from NETCDF file
    read_variables(&(u_k->h_var), &(v_k->h_var), &(e_k->h_var), &(tracer_k->h_var), &(particles->h_part_pos_x), &(particles->h_part_pos_y), &(H_xy->h_var));
    // Copy the initial conditions to the GPU
    cudaMemcpy(e_k->d_var, e_k->h_var, Nx*Ny*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u_k->d_var, u_k->h_var, Nx*Ny*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_k->d_var, v_k->h_var, Nx*Ny*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(tracer_k->d_var, tracer_k->h_var, Nx*Ny*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(particles->d_part_pos_x, particles->h_part_pos_x, num_particles*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(particles->d_part_pos_y, particles->h_part_pos_y, num_particles*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(H_xy->d_var, H_xy->h_var, Nx*Ny*sizeof(double), cudaMemcpyHostToDevice);
    // // Initialize the particles on the GPU
    // ParticleTi<<<1+((particles->num_particles-1)/tpb),tpb>>>(particles->d_part_pos_x, particles->d_part_pos_y, particles->num_particles, Lx, Ly);
    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );
    // Take derivatives of ICs
    e_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    u_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    v_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    tracer_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);

    H_xy->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);

    // Print
    Print_To_File(cntr);
};

Shallow_Water_Eq::~Shallow_Water_Eq () {
    cudaFree(d_k);
    cudaFree(d_l);
    cudaFree(d_filter);
    //free(h_k);
    //free(h_l);
    free(h_filter);
};


struct grid_index : public thrust::binary_function<double,double,int>
//Thrust function to calculate the bucket grid index based of a particle's positional coordinates
{
    const double gs_x;
    const double gs_y;
    const int nx_intgrid;

    grid_index(double _gs_x,double _gs_y, int _nx_intgrid) : gs_x(_gs_x),gs_y(_gs_y),nx_intgrid(_nx_intgrid) {}

    __host__ __device__
    int operator()(const double& x, const double& y) const {
                    return int(x/gs_x) + nx_intgrid*int(y/gs_y);
                    }
};



typedef thrust::tuple<int, int> Tuple;

struct TupleComp
{
    __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
    {
        if (t1.get<0>() < t2.get<0>())
            return true;
        if (t1.get<0>() > t2.get<0>())
            return false;
        return t1.get<1>() < t2.get<1>();
    }
};


void Shallow_Water_Eq::writeInteractions (const int cntr) {
//h represents host and d repesents device.    
    
    double *h_box_size;
    double *d_box_size;// Domain size

    int *h_grid_dim;
    int *d_grid_dim;// Number of buckets

    double *d_part_pos_x_sorted,*d_part_pos_y_sorted; // copy of arrays for sorting 
    int *d_grid_index,*d_atom_index,*d_num_interactions,*d_start_interactions;
    int *d_storepair1,*d_storepair2;
    int *h_storepair1,*h_storepair2;//Lists for storing the interaction pairs.

    size_t memsize_index,memsize_interactions,memsize_grid;
    int nx_intgrid,ny_intgrid;
    double gs_x,gs_y;
    double r; // temporary definition

    int blockSize, nBlocks;

    int ncid;
    char FILE_NAME[24];
    int atomindid,pair1_id,pair2_id;
    int FLAG = NC_NETCDF4;

    memsize_index= particles->num_particles * sizeof(int);
//The interaction distance read from the initial conditions file.
    r = parint; 

    h_box_size = (double *)malloc(2*sizeof(double));
    h_grid_dim = (int *)malloc(2*sizeof(int));

    h_box_size[0]=Lx;
    h_box_size[1]=Ly;

    nx_intgrid=h_box_size[0]/r;
    ny_intgrid=h_box_size[1]/r;
    gs_x=h_box_size[0]/nx_intgrid;
    gs_y=h_box_size[1]/ny_intgrid;
    h_grid_dim[0]=nx_intgrid;
    h_grid_dim[1]=ny_intgrid;
   
//Allocate device memory to variables and create thrust device pointers
    cudaMalloc((void **) &d_part_pos_x_sorted, particles->num_particles*sizeof(double));
    cudaMalloc((void **) &d_part_pos_y_sorted, particles->num_particles*sizeof(double)); 
    thrust::device_ptr<double> d_ptr_x_sorted(d_part_pos_x_sorted);
    thrust::device_ptr<double> d_ptr_y_sorted(d_part_pos_y_sorted);

    thrust::device_ptr<double> d_ptr_x_original(particles->d_part_pos_x);
    thrust::device_ptr<double> d_ptr_y_original(particles->d_part_pos_y);

    thrust::copy(thrust::device, d_ptr_x_original, d_ptr_x_original+particles->num_particles, d_ptr_x_sorted);
    thrust::copy(thrust::device, d_ptr_y_original, d_ptr_y_original+particles->num_particles, d_ptr_y_sorted);

    cudaMalloc((void **) &d_box_size, 2*sizeof(double));
    cudaMalloc((void **) &d_grid_dim, 2*sizeof(int));
    cudaMalloc((void **) &d_grid_index, memsize_index);
    cudaMalloc((void **) &d_atom_index, memsize_index);
    cudaMalloc((void **) &d_num_interactions, memsize_index);
    cudaMalloc((void **) &d_start_interactions, memsize_index);


    cudaMemcpy(d_box_size, h_box_size, 2*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_dim, h_grid_dim, 2*sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_ptr<int> d_ptr_grid_index(d_grid_index);
    thrust::device_ptr<int> d_ptr_atom_index(d_atom_index);
    thrust::device_ptr<int> d_ptr_num_interactions(d_num_interactions);
    thrust::device_ptr<int> d_ptr_start_interactions(d_start_interactions);

// fill d_atom_index with 0,1,2,3,...
    thrust::sequence(thrust::device, d_ptr_atom_index, d_ptr_atom_index+particles->num_particles);

// Find out the bucket index for each particle/atom by using the grid_index function.
    thrust::transform(d_ptr_x_sorted, d_ptr_x_sorted+particles->num_particles, d_ptr_y_sorted, d_ptr_grid_index, grid_index(gs_x,gs_y,nx_intgrid));

//Create temporary keys and copy the bucket grid index onto them.
    thrust::device_vector<int> tmpkey1(particles->num_particles);
    thrust::copy(thrust::device, d_ptr_grid_index, d_ptr_grid_index+particles->num_particles, tmpkey1.begin());

    thrust::device_vector<int> tmpkey2(particles->num_particles);
    thrust::copy(thrust::device, d_ptr_grid_index, d_ptr_grid_index+particles->num_particles, tmpkey2.begin());

//Sort the x and y positions of the particles according to the bucket indices  
    thrust::sort_by_key(tmpkey1.begin(),tmpkey1.end(),d_ptr_x_sorted);
    thrust::sort_by_key(tmpkey2.begin(),tmpkey2.end(),d_ptr_y_sorted);
//Sort the bucket grid indices based on the particle indices
    thrust::sort_by_key(d_ptr_grid_index,d_ptr_grid_index+particles->num_particles,d_ptr_atom_index);

// find the beginning of each bucket's list of points

    int *d_bucket_begin,*d_bucket_end;

    memsize_grid= nx_intgrid*ny_intgrid * sizeof(int);

    cudaMalloc((void **) &d_bucket_begin, memsize_grid);
    cudaMalloc((void **) &d_bucket_end, memsize_grid);

    thrust::device_ptr<int> d_ptr_bucket_begin(d_bucket_begin);
    thrust::device_ptr<int> d_ptr_bucket_end(d_bucket_end);

    thrust::counting_iterator<unsigned int> search_begin(0);

// find the beginning of each bucket's list of points
    thrust::lower_bound(d_ptr_grid_index,
        d_ptr_grid_index+particles->num_particles,
        search_begin,
        search_begin + nx_intgrid*ny_intgrid,
        d_ptr_bucket_begin);

// find the end of each bucket's list of points
    thrust::upper_bound(d_ptr_grid_index,
        d_ptr_grid_index+particles->num_particles,
        search_begin,
        search_begin + nx_intgrid*ny_intgrid,
        d_ptr_bucket_end);

// detect number of interactions for each particle/atom, store in array
    d_storepair1=NULL;
    d_storepair2=NULL;

    blockSize = 512;
    nBlocks = particles->num_particles / blockSize + (particles->num_particles % blockSize > 0);

//Call interactions_grid first time to detect the number of interactions per particle
    interactions_grid<<<nBlocks, blockSize>>>(d_part_pos_x_sorted, d_part_pos_y_sorted, d_grid_dim, 
		    d_bucket_begin, d_bucket_end, d_box_size, d_atom_index, d_num_interactions, d_start_interactions, 
		    r ,particles->num_particles,0,d_storepair1,d_storepair2);

// all atom kernel for debugging
//    interactions_n2<<<nBlocks, blockSize>>>(d_part_pos_x_sorted, d_part_pos_y_sorted, 
//                    d_box_size, d_atom_index, d_num_interactions, d_start_interactions,
//                    r ,particles->num_particles,0,d_storepair1,d_storepair2);

//exclusive_scan gives us position of each particle in the storing lists, based on the number of interactions each particle has.
    thrust::exclusive_scan(thrust::device, d_ptr_num_interactions, d_ptr_num_interactions + particles->num_particles, d_ptr_start_interactions);

 // obtain size of array to allocate for pair storage
    int n1,n2,ninteractions;
    cudaMemcpy(&n1, d_num_interactions+particles->num_particles-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n2, d_start_interactions+particles->num_particles-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
//A way of determining the size of our storing list, obtained by summing the number of interactions of the last particle and the it's start position in the storing list.   
    ninteractions=n1+n2;
    memsize_interactions=ninteractions*sizeof(int);

// allocate array to store list of interactions
// as the number of interactions unknown a priori, will check cudaMalloc for errors

    gpuErrchk( cudaMalloc((void **) &d_storepair1, memsize_interactions) );
    gpuErrchk( cudaMalloc((void **) &d_storepair2, memsize_interactions) );

//Call interactions_grid to compute the storing lists.
    interactions_grid<<<nBlocks, blockSize>>>(d_part_pos_x_sorted, d_part_pos_y_sorted, d_grid_dim, 
		    d_bucket_begin, d_bucket_end, d_box_size, d_atom_index, d_num_interactions, d_start_interactions, 
		    r ,particles->num_particles,1,d_storepair1,d_storepair2);

// all atom kernel for debugging
//    interactions_n2<<<nBlocks, blockSize>>>(d_part_pos_x_sorted, d_part_pos_y_sorted, 
//                    d_box_size, d_atom_index, d_num_interactions, d_start_interactions,
//                    r ,particles->num_particles,1,d_storepair1,d_storepair2);

// sort results, as tuple, by first index and then second index

    thrust::device_ptr<int> d_ptr_storepair1(d_storepair1);
    thrust::device_ptr<int> d_ptr_storepair2(d_storepair2);

    thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(d_ptr_storepair1, d_ptr_storepair2)), thrust::make_zip_iterator(thrust::make_tuple(d_ptr_storepair1 + ninteractions, d_ptr_storepair2 + ninteractions)), TupleComp());


// allocate host arrays to store pair data
    h_storepair1 = (int *)malloc(memsize_interactions);
    h_storepair2 = (int *)malloc(memsize_interactions);

    cudaMemcpy(h_storepair1, d_storepair1, memsize_interactions, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_storepair2, d_storepair2, memsize_interactions, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    sprintf(FILE_NAME, "interactions_%d.nc",cntr);
    // Create file
    nc_create(FILE_NAME, FLAG, &ncid);

    nc_def_dim(ncid, "Atomindex", ninteractions, &atomindid);

    nc_def_var(ncid, "Pair1", NC_INT, 1, &atomindid, &pair1_id);
    nc_put_var_int(ncid, pair1_id, h_storepair1);

    nc_def_var(ncid, "Pair2", NC_INT, 1, &atomindid, &pair2_id);
    nc_put_var_int(ncid, pair2_id, h_storepair2);

    nc_close(ncid);

    cudaFree(d_part_pos_x_sorted);
    cudaFree(d_part_pos_y_sorted);
    cudaFree(d_box_size);
    cudaFree(d_grid_dim);
    cudaFree(d_grid_index);
    cudaFree(d_atom_index);
    cudaFree(d_num_interactions);
    cudaFree(d_start_interactions);
    cudaFree(d_bucket_begin);
    cudaFree(d_bucket_end);
    cudaFree(d_storepair1);
    cudaFree(d_storepair2);


    free(h_storepair1);
    free(h_storepair2);
    free(h_box_size);
    free(h_grid_dim);
}


void Shallow_Water_Eq::Print_To_File (const int cntr) {
    // Copy the variables that are going to be printed back to the host
    cudaMemcpy(e_k->h_var, e_k->d_var, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_k->h_var, u_k->d_var, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_k->h_var, v_k->d_var, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(tracer_k->h_var, tracer_k->d_var, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u_k->h_vary, u_k->d_vary, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_k->h_varx, v_k->d_varx, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles->h_part_pos_x, particles->d_part_pos_x, particles->num_particles*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles->h_part_pos_y, particles->d_part_pos_y, particles->num_particles*sizeof(double), cudaMemcpyDeviceToHost);
    doEnergy();
    doVort();

    char FILE_NAME[24];
    sprintf(FILE_NAME, "output_%d.nc",cntr);
    // int retval;
    int const ndims = 3;
    // ID variables
    int ncid, timedimid, xdimid, ydimid, trajdimid, timevarid, trajvarid, xvarid, yvarid, uvarid, vvarid, etavarid, tracervarid, vortvarid,
        particlexvarid, particleyvarid, kevarid, apevarid;
    int constid, output_varid, Nx_varid, Ny_varid, num_particles_varid, g_varid, H_varid, f_varid, Lx_varid, Ly_varid, Tf_varid, plot_interval_varid;
    int dimids[ndims];
    int partdimids[2];
    int FLAG = NC_NETCDF4;

    // Create file
    nc_create(FILE_NAME, FLAG, &ncid);
        // ERR(retval);

    // Define dimensions
    nc_def_dim(ncid, "time", 1, &timedimid);

    nc_def_dim(ncid, "X", Nx, &xdimid);
        // ERR(retval);
    nc_def_dim(ncid, "Y", Ny, &ydimid);
        // ERR(retval);
    nc_def_dim(ncid, "trajectory", particles->num_particles, &trajdimid);

    nc_def_dim(ncid, "const", 1, &constid);

    // Fill dimensions and constants
    nc_def_var(ncid, "time", NC_DOUBLE, 0, &timedimid, &timevarid);

    nc_def_var(ncid, "X", NC_DOUBLE, 1, &xdimid, &xvarid);
        // ERR(retval);
    nc_def_var(ncid, "Y", NC_DOUBLE, 1, &ydimid, &yvarid);
        // ERR(retval);
    nc_def_var(ncid, "trajectory", NC_DOUBLE, 1, &trajdimid, &trajvarid);

    nc_def_var(ncid, "Nx", NC_INT, 0, &constid, &Nx_varid);

    nc_def_var(ncid, "Ny", NC_INT, 0, &constid, &Ny_varid);

    nc_def_var(ncid, "num_particles", NC_INT, 0, &constid, &num_particles_varid);

    nc_def_var(ncid, "output", NC_INT, 0, &constid, &output_varid);

    nc_def_var(ncid, "g", NC_DOUBLE, 0, &constid, &g_varid);

    nc_def_var(ncid, "Hmax", NC_DOUBLE, 0, &constid, &H_varid);

    nc_def_var(ncid, "f", NC_DOUBLE, 0, &constid, &f_varid);

    nc_def_var(ncid, "Lx", NC_DOUBLE, 0, &constid, &Lx_varid);

    nc_def_var(ncid, "Ly", NC_DOUBLE, 0, &constid, &Ly_varid);

    nc_def_var(ncid, "Tf", NC_DOUBLE, 0, &constid, &Tf_varid);

    nc_def_var(ncid, "plot_interval", NC_DOUBLE, 0, &constid, &plot_interval_varid);

    dimids[0] = timedimid;
    dimids[1] = xdimid;
    dimids[2] = ydimid;
    partdimids[0] = timedimid;
    partdimids[1] = trajdimid;

    // Fill variables
    nc_def_var(ncid, "u", NC_DOUBLE, ndims, dimids, &uvarid);
        // ERR(retval);
    nc_def_var(ncid, "v", NC_DOUBLE, ndims, dimids, &vvarid);
        // ERR(retval);
    nc_def_var(ncid, "eta", NC_DOUBLE, ndims, dimids, &etavarid);
        // ERR(retval);
    nc_def_var(ncid, "tracer", NC_DOUBLE, ndims, dimids, &tracervarid);

    nc_def_var(ncid, "vorticity", NC_DOUBLE, ndims, dimids, &vortvarid);
        // ERR(retval);
    nc_def_var(ncid, "particle_x_position", NC_DOUBLE, 2, partdimids, &particlexvarid);
        // ERR(retval);
    nc_def_var(ncid, "particle_y_position", NC_DOUBLE, 2, partdimids, &particleyvarid);
        // ERR(retval);
    nc_def_var(ncid, "total KE", NC_DOUBLE, 0, &timedimid, &kevarid);

    nc_def_var(ncid, "total APE", NC_DOUBLE, 0, &timedimid, &apevarid);

    nc_enddef(ncid);
        // ERR(retval);

    nc_put_var_double(ncid, timevarid, &tk);

    nc_put_var_double(ncid, xvarid, xgrid);
        // ERR(retval);
    nc_put_var_double(ncid, yvarid, ygrid);
        // ERR(retval);
    nc_put_var_double(ncid, trajvarid, trajectory);

    nc_put_var_double(ncid,uvarid,u_k->h_var);
        // ERR(retval);
    nc_put_var_double(ncid,vvarid,v_k->h_var);
        // ERR(retval);
    nc_put_var_double(ncid,etavarid,e_k->h_var);
        // ERR(retval);
    nc_put_var_double(ncid,tracervarid,tracer_k->h_var);

    nc_put_var_double(ncid,vortvarid,vort);
        // ERR(retval);
    nc_put_var_double(ncid,particlexvarid,particles->h_part_pos_x);
        // ERR(retval);
    nc_put_var_double(ncid,particleyvarid,particles->h_part_pos_y);
        // ERR(retval);
    nc_put_var_double(ncid,kevarid,&total_ke);

    nc_put_var_double(ncid,apevarid,&total_pe);

    nc_put_var_int(ncid, Nx_varid, &Nx);

    nc_put_var_int(ncid, Ny_varid, &Ny);

    nc_put_var_int(ncid, num_particles_varid, &num_particles);

    nc_put_var_int(ncid, output_varid, &cntr);

    nc_put_var_double(ncid, g_varid, &g);

    nc_put_var_double(ncid, H_varid, &Hmax);

    nc_put_var_double(ncid, f_varid, &f);

    nc_put_var_double(ncid, Lx_varid, &Lx);

    nc_put_var_double(ncid, Ly_varid, &Ly);

    nc_put_var_double(ncid, Tf_varid, &Tf);

    nc_put_var_double(ncid, plot_interval_varid, &plot_interval);

    nc_close(ncid);
}

void Shallow_Water_Eq::doDerivatives () {
    // Compute derivatives
    e_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    u_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    v_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    H_xy->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    tracer_k->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
}

void Shallow_Water_Eq::doRK2() {
    // Start by filtering the variables
    e_k->Filter(d_filter, Nx, Ny, tpb, nblks);
    u_k->Filter(d_filter, Nx, Ny, tpb, nblks);
    v_k->Filter(d_filter, Nx, Ny, tpb, nblks);
    tracer_k->Filter(d_filter, Nx, Ny, tpb, nblks);
    // Take an Euler step
    EulerStep<<<nblks,tpb>>>(e_j->d_var, e_k->d_var, e_k->d_varx, e_k->d_vary,
                            u_j->d_var, u_k->d_var, u_k->d_varx, u_k->d_vary,
                            v_j->d_var, v_k->d_var, v_k->d_varx, v_k->d_vary,
			    tracer_j->d_var, tracer_k->d_var, tracer_k->d_varx, tracer_k->d_vary,
                            H_xy->d_var, H_xy->d_varx, H_xy->d_vary,
			    Nx, Ny, dtk, f, g);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // Compute derivatives at the Euler step
    e_j->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    u_j->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    v_j->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    H_xy->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    tracer_j->ComputeDerivatives(d_k, d_l, Nx, Ny, tpb, nblks);
    // Take the RK2 step and write to original (k) variable
    RK2<<<nblks,tpb>>>(     e_j->d_var,      e_j->d_varx,      e_j->d_vary,      e_k->d_var,      e_k->d_varx,      e_k->d_vary,
                            u_j->d_var,      u_j->d_varx,      u_j->d_vary,      u_k->d_var,      u_k->d_varx,      u_k->d_vary,
                            v_j->d_var,      v_j->d_varx,      v_j->d_vary,      v_k->d_var,      v_k->d_varx,      v_k->d_vary,
                            tracer_j->d_var, tracer_j->d_varx, tracer_j->d_vary, tracer_k->d_var, tracer_k->d_varx, tracer_k->d_vary,
                            H_xy->d_var,H_xy->d_varx,H_xy->d_vary,
			    Nx, Ny, dtk, f, g);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    doDerivatives();
    // Step time and calculate new dt
    tk = tk + dtk;
    dtk = adaptive_timestep();
}

void Shallow_Water_Eq::doParticle () {
    // Execute the particle interpolation and timestepper
    particle_interp_evolve_better<<<1+((particles->num_particles-1)/tpb),tpb>>>(u_k->d_var, v_k->d_var, particles->d_part_pos_x,
                                                                                particles->d_part_pos_y, particles->interp_u, 
                                                                                particles->interp_v, Nx, Ny, particles->num_particles, 
                                                                                dtk, dx, dy);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

double Shallow_Water_Eq::adaptive_timestep () {
    // Calculate the new dt based on 10% of the cfl condition
    return fmin(dx/max(u_k->MaxVar(Nx, Ny),c0),  dy/max(v_k->MaxVar(Nx, Ny),c0) ) /10.;
}

void Shallow_Water_Eq::doEnergy () {
    double* tmp;
    tmp = new double[Nx*Ny];
    for (int ii=0; ii<Nx*Ny; ii++) {
        tmp[ii] = H_xy->h_var[ii]*(u_k->h_var[ii]*u_k->h_var[ii] + v_k->h_var[ii]*v_k->h_var[ii]);
    }
    total_ke = 0.5 * std::accumulate(tmp, tmp + Nx*Ny, 0.0); 
    for (int ii=0; ii<Nx*Ny; ii++) {
        tmp[ii] = e_k->h_var[ii]*e_k->h_var[ii];
    }
    total_pe = 0.5 * g * std::accumulate(tmp, tmp + Nx*Ny, 0.0);
}

void Shallow_Water_Eq::doVort () {
    for (int ii=0; ii<Nx*Ny; ii++) {
        vort[ii] = v_k->h_varx[ii]-u_k->h_vary[ii];
    }
}

void Shallow_Water_Eq::read_parameters(int* Nx, int* Ny, int* num_particles, double* parint, int* cntr, double* Hmax, double* g, double* f, double* Lx, double* Ly, double* Ti, double* Tf, double* plot_interval) {
    // Open the NETCDF file
    int FLAG = NC_NOWRITE;
    int ncid=0;
    nc_open("initial_conditions.nc", FLAG, &ncid);
    int Nx_varid;
    nc_inq_varid(ncid, "Nx", &Nx_varid);
    nc_get_var_int(ncid, Nx_varid, Nx);
    int Ny_varid;
    nc_inq_varid(ncid, "Ny", &Ny_varid);
    nc_get_var_int(ncid, Ny_varid, Ny);
    int num_particles_varid;
    nc_inq_varid(ncid, "num_particles", &num_particles_varid);
    nc_get_var_int(ncid, num_particles_varid, num_particles);
    int output_varid;
    nc_inq_varid(ncid, "output", &output_varid);
    nc_get_var_int(ncid, output_varid, cntr);
    int g_varid;
    nc_inq_varid(ncid, "g", &g_varid);
    nc_get_var_double(ncid, g_varid, g);
    int H_varid;
    nc_inq_varid(ncid, "Hmax", &H_varid);
    nc_get_var_double(ncid, H_varid, Hmax);
    int parint_varid;
    nc_inq_varid(ncid, "par_int", &parint_varid);
    nc_get_var_double(ncid, parint_varid, parint);
    int f_varid;
    nc_inq_varid(ncid, "f", &f_varid);
    nc_get_var_double(ncid, f_varid, f);
    int Lx_varid;
    nc_inq_varid(ncid, "Lx", &Lx_varid);
    nc_get_var_double(ncid, Lx_varid, Lx);
    int Ly_varid;
    nc_inq_varid(ncid, "Ly", &Ly_varid);
    nc_get_var_double(ncid, Ly_varid, Ly);
    int time_varid;
    nc_inq_varid(ncid, "time", &time_varid);
    nc_get_var_double(ncid, time_varid, Ti);
    int Tf_varid;
    nc_inq_varid(ncid, "Tf", &Tf_varid);
    nc_get_var_double(ncid, Tf_varid, Tf);
    int plot_interval_varid;
    nc_inq_varid(ncid, "plot_interval", &plot_interval_varid);
    nc_get_var_double(ncid, plot_interval_varid, plot_interval);
}


void Shallow_Water_Eq::read_variables(double** my_u, double** my_v, double** my_eta, double** my_t, double** my_xp, double** my_yp, double** my_Hxy) {
    // Open the NETCDF file
    int FLAG = NC_NOWRITE;
    int ncid=0;
    nc_open("initial_conditions.nc", FLAG, &ncid);
    // Declare variables
    int u_varid, v_varid, eta_varid, t_varid, xp_varid, yp_varid, Hxy_varid;
    nc_inq_varid(ncid, "u", &u_varid);
    nc_inq_varid(ncid, "v", &v_varid);
    nc_inq_varid(ncid, "eta", &eta_varid);
    nc_inq_varid(ncid, "tracer", &t_varid);
    nc_inq_varid(ncid, "H", &Hxy_varid);
    nc_inq_varid(ncid, "particle_x_position", &xp_varid);
    nc_inq_varid(ncid, "particle_y_position", &yp_varid);
    nc_get_var_double(ncid, u_varid, my_u[0]);
    nc_get_var_double(ncid, v_varid, my_v[0]);
    nc_get_var_double(ncid, eta_varid, my_eta[0]);
    nc_get_var_double(ncid, t_varid, my_t[0]);
    nc_get_var_double(ncid, Hxy_varid, my_Hxy[0]);
    nc_get_var_double(ncid, xp_varid, my_xp[0]);
    nc_get_var_double(ncid, yp_varid, my_yp[0]);
}
