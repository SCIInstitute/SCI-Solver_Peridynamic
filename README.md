# PeriCUDA

A CUDA-accelerated peridynamics simulation.

Copyright (c) 2016 Nghia Truong
This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

The project has been deprecated. There are still many bugs and performance issues.

SPH simulation has been implemented but it is not completed. It may work, but incorrectly.


# Build
```
cd <PATH>/PeriCUDAEngine
make -j

cd <PATH>/BunnyBreak/mesh_query0.1
make -j

cd <PATH>/BunnyBreak
make -j
```

# Run:
```
cd <PATH>/BunnyBreak
./BunnyBreak params.txt
```

# Parameters
```
adaptive_integration = 0/1: integration using adaptive time step or fixed time step
num_pd_particle = 0/<non-zero>: if it is zero, no particle generated. non zero: generated particles to fill a mesh
pd_stretch_limit_s0 = <double>: set threshold on bond stretch. Bonds that stretched beyond this threshold will break
mesh_translation_x/y/z: translate the mesh in domain
steps_per_frame = <int>: save data every <steps_per_frame> time steps
final_step = <int>: final time step
obj_file = <string>: mesh file. The mesh should be watertight
saving_path = <string>: where data will be saved
boundary_min/max_x/y/z: simulation domain
pd_kernel_coeff = <double>: define the radius of particle connection. For example, pd_kernel_coeff=5 means that a particle will connect to other particles that stay within 5 times particle radius
pd_particle_mass = <double>: mass of particle

Other sph_ parameters are used for sph simulation.
```

# Data Visualization

The visualization software is written in Qt 5.6.

After having simulation data, select data path(or press B) and browse to folder SAVED_DATA_FOLDER/DATA.
