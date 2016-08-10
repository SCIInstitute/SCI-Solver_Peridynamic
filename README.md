SCI-Solver_Peridynamic
======================

SCI-Solver_Peridynamic is a C++/CUDA library written to simulate a fracture behavior in a particle cloud model. It uses GPU hardware to provide a fast solution.

The code was written by Nghia Truong at the Scientific Computing and Imaging Institute, 
University of Utah, Salt Lake City, USA.
<br/>
![alt tag](https://raw.githubusercontent.com/SCIInstitute/SCI-Solver_Peridynamic/master/src/Resources/figure6-1.png)
Table of Contents
========
- [Acknowledgements](#acknowledgements)
- [Requirements](#requirements)
- [Building](#building)<br/>
		- [Linux / OSX](#linux-and-osx)<br/>
		- [Windows](#windows)<br/>
- [Running Examples](#running-examples)
- [Using the Library](#using-the-library)

Acknowledgements
=========
**<a href ="http://sealab.cs.utah.edu/Papers/Levine-2014-APP/">A Peridynamic Perspective on Spring-Mass Fracture</a>**<br/>

**AUTHORS:**
<br/>J. A. Levine(*1*) <br/>
A. W. Bargteil(*2*) <br/>
C. Corsi(*1*) <br/>
J. Tessendorf(*1*) <br/>
R. Geist(*1*) <br/>
(*1*)Clemson University, USA  (*2*)University of Utah, USA

The  application  of  spring-mass  systems  to  the  animation  of  brittle  fracture  is  revisited.  The  motivation  arisesfrom the recent popularity ofperidynamicsin the computational physics community. Peridynamic systems can beregarded as spring-mass systems with two specific properties. First, spring forces are based on a simple strainmetric, thereby decoupling spring stiffness from spring length. Second, masses are connected using a distance-based criterion. The relatively large radius of influence typically leads to a few hundred springs for every masspoint. Spring-mass systems with these properties are shown to be simple to implement, trivially parallelized, andwell-suited to animating brittle fracture.
<br/><br/>
Requirements
==============

 * Git and the standard system build environment tools.
 * You will need a CUDA Compatible Graphics card. See <a href="https://developer.nvidia.com/cuda-gpus">here</a> You will also need to be sure your card has CUDA compute capability of at least 2.0.
 * SCI-Solver_Peridynamic is compatible with the latest CUDA toolkit (7.5). Download <a href="https://developer.nvidia.com/cuda-downloads">here</a>.
 * This project has been tested on Ubuntu 14.04 on NVidia GeForce GTX 560 Ti, and OSX 10.11 on NVidia GeForce GTX 780M. 
 * If you have a CUDA graphics card equal to or greater than our test machines and are experiencing issues, please contact the repository owners.
 * OSX: Please be sure to follow setup for CUDA <a href="http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-mac-os-x/#axzz3W4nXNNin">here</a>. There are several compatability requirements for different MAC machines, including using a different version of CUDA (ie. 5.5).
 * Simulation visualizer uses QtCreator >= 5.6 to view results (optional).

Building
==============

<h3>Linux and OSX</h3>
Makefiles are configured to use the gcc compiler for a linux environment, and the clang compiler in Mac OSX. There are two static libraries and one example to build. The components and their Makefiles are as follows:
 * PeriCUDA Engine:  PeriCUDAEngine/Makefile
 * Mesh Query:  BunnyBreak/mesh_query0.1/Makefile
 * BunnyBreak example: BunnyBreak/Makefile
The CUDA Compute capability is specified in all Makefiles using the compiler flag string:
```c++
arch=compute_20,code=sm_20
```
where 20 represents the major (2) and minor (.0) numbers to specify a compute capability of 2.0. Terminal commands for building the three libraries:
```c++
cd <PATH>/PeriCUDAEngine
make -j

cd <PATH>/BunnyBreak/mesh_query0.1
make -j

cd <PATH>/BunnyBreak
make -j
```

<h3>Windows</h3>
This software has not yet been ported to Windows.

**Note:** For all platforms, you may need to specify your CUDA toolkit location (especially if you have multiple CUDA versions installed):
```c++
cmake -DCUDA_TOOLKIT_ROOT_DIR="~/NVIDIA/CUDA-7.5" ../src
```
(Assuming this is the location).

**Note:** If you have compile errors such as <code>undefined reference: atomicAdd</code>, it is likely you need to set your compute capability manually.

Running Examples
==============

The "BunnyBreak" example is run from <PATH>/BunnyBreak. The params.txt file is used to specify the necessary parameters for the simulation run. Use a text editor to set the following two paths to suit your system:
 * obj_file /path/to/repo/PeriCUDA/BunnyBreak/bunny.obj
 * saving_path /path/to/repo/PeriCUDA/BunnyBreak/SimData/BunnyBreak
Other parameters in the file can also be modified, including the following (with descriptions):
 * adaptive_integration = 0/1: integration using adaptive time step or fixed time step
 * num_pd_particle = 0/<non-zero>: if it is zero, no particle generated. non zero: generated particles to fill a mesh
 * pd_stretch_limit_s0 = <double>: set threshold on bond stretch. Bonds that stretched beyond this threshold will break
 * mesh_translation_x/y/z: translate the mesh in domain
 * steps_per_frame = <int>: save data every <steps_per_frame> time steps
 * final_step = <int>: final time step
 * obj_file = <string>: mesh file. The mesh should be watertight
 * saving_path = <string>: where data will be saved
 * boundary_min/max_x/y/z: simulation domain
 * pd_kernel_coeff = <double>: define the radius of particle connection. For example, pd_kernel_coeff=5 means that a particle will connect to other particles that stay within 5 times particle radius
 * pd_particle_mass = <double>: mass of particle

To run the simulation, type:
```c++
cd <PATH>/BunnyBreak
./BunnyBreak params.txt
```

To visualize the results, start QtCreator, open the project file <PATH>/PeriCUDAViz/PeriCUDAViz.pro & browse to the DATA/ sub-directory of the *saving_path* specified in the params.txt file.

Using the Library
==============

The BunnyBreak project provides a basic usage example. The project is built by including the static library SPHPEngine.a, and the main.cpp and scene.cpp files. In main.cpp, instances of the Scene and Simulator objects are instantiated, the parameters are set according to the information specified in params.txt, and the simulator is run.
<br/><br/>
*Original Author's Note:* This project has been discontinued in favor of CPU simulations that handle much larger models. SPH simulation has been implemented but not completed, so it may work incorrectly.
