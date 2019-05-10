# CUDA_C-_SW_particle_code

This project solves the Shallow Water Equations using CUDA. 
This also does dynamic particle tracking, where particles are Lagrangian objections advected by the flow.
Shallow_Water_Eq.cu has a method writeInteractions() which can calculate inter-particle distances for a million particles
using GPUs.
All the GPU compuatations are done in cuda_kernels.cu. 
The documentation for computing particle interactions using cuda is provided as __Cuda_compute_particle_interactions_doc.pdf__ .

The .py files are needed to create initial conditions.

To run the code:

make clean
python3 **.py
make
./SW_main.x

Outputs and interactions are saved as .nc files


