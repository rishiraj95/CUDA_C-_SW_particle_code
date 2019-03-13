from netCDF4 import Dataset
import numpy as np
import sys

##
### Specify simulation parameters
##

# Parallelization parameters
Nproc_x         = 1
Nproc_y         = 1
Nthreads_solver = 4

Nproc_tracker    = 1
Nthreads_tracker = 4

# Simulation paramters

Nx = 1024
Ny = 1024

Lx = 10.
Ly = 10.

dx = Lx/Nx
dy = Ly/Ny

gravity = 9.81
H0      = 0.5
f0      = 1.
visco   = 0.01 * np.sqrt(gravity * H0) * max(dx, dy)

final_time = 100.
num_outs   = 100

num_particles = 500**2
max_inertia = 1.
min_inertia = 0.

# Create the grid
x = np.arange(dx/2., Lx, dx)
y = np.arange(dy/2., Ly, dy)

X, Y = np.meshgrid(x, y)

Y -= y.mean()
X -= x.mean()

# Specify initial conditions
a0 = 0.1 * H0
R  = np.sqrt(X**2 + Y**2)
W  = Ly / 10.
Wr = Ly / 5.

u_data = 2 * gravity * a0 / ( f0 * W ) * np.tanh( Y / W ) / ( ( np.cosh( Y / W ) )**2 )
v_data = 0 * X
h_data = H0 + ( a0 / ( ( np.cosh( Y / W ) )**2 ) ) + ( a0 / ( ( np.cosh( R / Wr ) )**8 ) )

# Specify initial particle positions

x_part = Lx * np.random.rand(num_particles)
#y_part = Ly * np.random.rand(num_particles)
#y_part = Ly/2. + W * 4 * ( np.random.rand(num_particles) - 0.5)
y_part = y[Ny/4] + (Ly/2.) * np.random.rand(num_particles)

inert_part = min_inertia + ( max_inertia - min_inertia )* np.random.rand(num_particles)

u_part = 2 * gravity * a0 / ( f0 * W ) * np.tanh( y_part / W ) / ( ( np.cosh( y_part / W ) )**2 )
v_part = 0*x_part

##
### Output to initial_conditions.nc
###   user shouldn't need to modify anything after this
##

# Create file
fp = Dataset('initial_conditions.nc', 'w', format='NETCDF4')
fp.description = 'Initial conditions for simulation'
    
# Create dimension objects
x_dim = fp.createDimension('x',Nx)
y_dim = fp.createDimension('y',Ny)
p_dim = fp.createDimension('particle_index',num_particles)
const_dim = fp.createDimension('const', 1)

# Create variables, assign attributes, and write data
x_grid = fp.createVariable('x','d',('x',))
x_grid.units = 'm'
x_grid[:] = x

y_grid = fp.createVariable('y','d',('y',))
y_grid.units = 'm'
y_grid[:] = y

p_grid = fp.createVariable('particle_index','d',('particle_index',))
p_grid.units = ''
p_grid[:] = np.arange(num_particles)

u_var = fp.createVariable('u', 'd', ('y','x'), contiguous=True)
u_var.units = 'm/s'
u_var[:,:] = u_data

v_var = fp.createVariable('v', 'd', ('y','x'), contiguous=True)
v_var.units = 'm/s'
v_var[:,:] = v_data

h_var = fp.createVariable('h', 'd', ('y','x'), contiguous=True)
h_var.units = 'm'
h_var[:,:] = h_data

xp_var = fp.createVariable('particles_x', 'd', ('particle_index'), contiguous=True)
xp_var.units = 'm'
xp_var[:] = x_part

yp_var = fp.createVariable('particles_y', 'd', ('particle_index'), contiguous=True)
yp_var.units = 'm'
yp_var[:] = y_part

ip_var = fp.createVariable('particles_inert', 'd', ('particle_index'), contiguous=True)
ip_var[:] = inert_part

up_var = fp.createVariable('particles_u', 'd', ('particle_index'), contiguous=True)
up_var.units = 'm/s'
up_var[:] = u_part

vp_var = fp.createVariable('particles_v', 'd', ('particle_index'), contiguous=True)
vp_var.units = 'm/s'
vp_var[:] = v_part

Nx_var = fp.createVariable('Nx', np.int, ('const',))
Nx_var[0] = Nx

Ny_var = fp.createVariable('Ny', np.int, ('const',))
Ny_var[0] = Ny

Lx_var = fp.createVariable('Lx', 'd', ('const',))
Lx_var[0] = Lx

Ly_var = fp.createVariable('Ly', 'd', ('const',))
Ly_var[0] = Ly

Nproc_x_var = fp.createVariable('Nproc_x', np.int, ('const',))
Nproc_x_var[0] = Nproc_x

Nproc_y_var = fp.createVariable('Nproc_y', np.int, ('const',))
Nproc_y_var[0] = Nproc_y

Nthreads_solver_var = fp.createVariable('Nthreads_solver', np.int, ('const',))
Nthreads_solver_var[0] = Nthreads_solver

Nproc_tracker_var = fp.createVariable('Nproc_tracker', np.int, ('const',))
Nproc_tracker_var[0] = Nproc_tracker

Nthreads_tracker_var = fp.createVariable('Nthreads_tracker', np.int, ('const',))
Nthreads_tracker_var[0] = Nthreads_tracker

gravity_var = fp.createVariable('gravity', 'd', ('const',))
gravity_var[0] = gravity

H0_var = fp.createVariable('H0', 'd', ('const',))
H0_var[0] = H0

f0_var = fp.createVariable('f0', 'd', ('const',))
f0_var[0] = f0

visco_var = fp.createVariable('visco', 'd', ('const',))
visco_var[0] = visco

final_time_var = fp.createVariable('final_time', 'd', ('const',))
final_time_var[0] = final_time

num_outs_var = fp.createVariable('num_outs', np.int, ('const',))
num_outs_var[0] = num_outs

num_particles_var = fp.createVariable('num_particles', np.int, ('const',))
num_particles_var[0] = num_particles

# Close the file
fp.close()

