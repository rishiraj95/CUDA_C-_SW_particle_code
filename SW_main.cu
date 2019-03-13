#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "Shallow_Water_Eq.h"
#include "Variable.h"
#include "Particle.h"
#include "cuda_kernels.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main () {
    // Set which GPU you want to run on
    // turn off on graham ::  gpuErrchk( cudaSetDevice(3) );
    //// Start Main Program ////
    printf("\nStarting Shallow Water Equation Solver\n\n\n");
    // Initialize
    printf("Initializing...\n");
    clock_t begin_initial = clock();
    Shallow_Water_Eq myCase (0);
    myCase.dtk = myCase.adaptive_timestep();
    myCase.cntr++;
    clock_t end_initial = clock();
    double duration_initial = (double)(end_initial-begin_initial) / CLOCKS_PER_SEC;
    printf("Time for initialization:\t%f seconds\n",duration_initial);
    printf("Done Initializing.\n");
    printf("Relavent Parameters:\nNx=%i\t\tNy=%i\nLx=%f\t\tLy=%f\nf=%f\nTf=%f\t\tplot_interval=%f\t\tintital_dt=%f\n"
        ,myCase.Nx,myCase.Ny,myCase.Lx,myCase.Ly,myCase.f,myCase.Tf,myCase.plot_interval,myCase.dtk);

    // // Main timestep
    printf("Entering main loop:\n\n");
    clock_t begin_loop = clock();
    while (myCase.tk < myCase.Tf) {
        clock_t step_begin = clock();
        myCase.doRK2();
        myCase.doParticle();
        myCase.timestep++;
        if (myCase.tk > myCase.next_output_time ) {
            myCase.Print_To_File (myCase.cntr);
            myCase.writeInteractions(myCase.cntr);
            printf("Output %u. \t%f%% Complete \t Time: %f\n",myCase.cntr,100*myCase.tk/myCase.Tf,myCase.tk);
            myCase.cntr++;
            myCase.next_output_time += myCase.plot_interval;
        }
    }
    clock_t end_loop = clock();
    double duration_loop = (double)(end_loop-begin_loop) / CLOCKS_PER_SEC;
    printf("Time for loop:\t%f seconds\n",duration_loop);
    printf("Done.\n");
    printf("Goodbye!\n\n");
}
