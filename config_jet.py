from netCDF4 import Dataset
import numpy as np
import sys

##
### Specify simulation parameters
##

# Simulation paramters

Nx = 256
Ny = 256

Lx = 10.
Ly = 10.

dx = Lx/Nx
dy = Ly/Ny

g       = 9.81
H0      = 0.5
f0      = 1.

output = 0
time  = 0.
final_time    = 20.
plot_interval = 1.

num_particles = 50**2

# Create the grid
x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)

X, Y = np.meshgrid(x, y)

Y -= y.mean()
X -= x.mean()

# Specify initial particle positions

x_p = np.linspace(0.0, Lx, np.sqrt(num_particles))
y_p = np.linspace(0.0, Ly, np.sqrt(num_particles))

Xp,Yp = np.meshgrid(x_p,y_p);

x_part = Xp.ravel()
y_part = Yp.ravel()

u_part = 0*x_part;
v_part = 0*x_part;

# Specify initial conditions


a0 = 0.1 * H0
R  = np.sqrt(X**2 + Y**2)
W  = Ly / 10.
Wr = Ly / 5.

u_data = 2 * g * a0 / ( f0 * W ) * np.tanh( Y / W ) / ( ( np.cosh( Y / W ) )**2 )
v_data = 0 * X
eta_data = ( a0 / ( ( np.cosh( Y / W ) )**2 ) ) + ( a0 / ( ( np.cosh( R / Wr ) )**8 ) )
t_data = np.zeros(X.shape)

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
traj_dim = fp.createDimension('trajectory',num_particles)
const_dim = fp.createDimension('const', 1)

# Create variables, assign attributes, and write data
u_var = fp.createVariable('u', 'd', ('y','x'), contiguous=True)
u_var.units = 'm/s'
u_var[:,:] = u_data

v_var = fp.createVariable('v', 'd', ('y','x'), contiguous=True)
v_var.units = 'm/s'
v_var[:,:] = v_data

eta_var = fp.createVariable('eta', 'd', ('y','x'), contiguous=True)
eta_var.units = 'm'
eta_var[:,:] = eta_data

t_var = fp.createVariable('tracer', 'd', ('y','x'), contiguous=True)
t_var.units = 'n/a'
t_var[:,:] = t_data

Nx_var = fp.createVariable('Nx', np.int, ('const',))
Nx_var[0] = Nx

Ny_var = fp.createVariable('Ny', np.int, ('const',))
Ny_var[0] = Ny

output_var = fp.createVariable('output', np.int, ('const',))
output_var[0] = output

Lx_var = fp.createVariable('Lx', 'd', ('const',))
Lx_var[0] = Lx

Ly_var = fp.createVariable('Ly', 'd', ('const',))
Ly_var[0] = Ly

g_var = fp.createVariable('g', 'd', ('const',))
g_var[0] = g

H0_var = fp.createVariable('H', 'd', ('const',))
H0_var[0] = H0

f0_var = fp.createVariable('f', 'd', ('const',))
f0_var[0] = f0

time_var = fp.createVariable('time', 'd', ('const',))
time_var[0] = time

Tf_var = fp.createVariable('Tf', 'd', ('const',))
Tf_var[0] = final_time

plot_interval_var = fp.createVariable('plot_interval', 'd', ('const',))
plot_interval_var[0] = plot_interval

num_particles_var = fp.createVariable('num_particles', np.int, ('const',))
num_particles_var[0] = num_particles

x_part_pos = fp.createVariable('particle_x_position', 'd', ('trajectory',))
x_part_pos[:] = x_part

y_part_pos = fp.createVariable('particle_y_position', 'd', ('trajectory',))
y_part_pos[:] = y_part

# Close the file
fp.close()

