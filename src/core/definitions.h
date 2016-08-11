//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <stdint.h>
#include <vector_types.h>

#define PRINT_DEBUG 0

////////////////////////////////////////////////////////////////////////////////
// single or double precision?
#define SINGLE_PRECISION 0

#if SINGLE_PRECISION
//typedef float real_t
//typedef float3 real3_t
//typedef float4 real4_t
#define real_t float
#define real3_t float3
#define real4_t float4

#define MAKE_REAL3(x, y, z) make_float3(x, y, z)
#define MAKE_REAL3_FROM_REAL(x) make_float3(x)
#define MAKE_REAL4(x, y, z, w) make_float4(x, y, z, w)
#define MAKE_REAL4_FROM_REAL(x) make_float4(x)
#define MAKE_REAL4_FROM_REAL3(x3, w) make_float4(x3, w)
#else // double precision
//typedef double real_t;
//typedef double3 real3_t;
//typedef double4 real4_t;
#define real_t double
#define real3_t double3
#define real4_t double4


#define MAKE_REAL3(x, y, z) make_double3(x, y, z)
#define MAKE_REAL3_FROM_REAL(x) make_double3(x)
#define MAKE_REAL3_FROM_REAL4(x) make_double3(x)
#define MAKE_REAL4(x, y, z, w) (make_double4(x, y, z, w))
#define MAKE_REAL4_FROM_REAL(x) make_double4(x)
#define MAKE_REAL4_FROM_REAL3(x3, w) make_double4(x3, w)
#endif


//#define CELL_SIZE (0.02*3.126)


#define NUM_CELL_X 32
#define NUM_CELL_Y 32
#define NUM_CELL_Z 64

#define NUM_CELLS (NUM_CELL_X * NUM_CELL_Y * NUM_CELL_Z)

#define INFTY (1.0e100)
#define SMALL (1.0e-10)
#define PI (3.14159265358979323846264)


#define SYSTEM_TIMESTEP_BASE (1.0/8388608.0)  // = 2^23

#define TIMESTEP_SPH 1024
#define TIMESTEP_PD (128)
#define TIMESTEP_PD_MIN 100

#define MAX_SUBSTEP 10


#define RESTITUTION 0.001
//#define RESTITUTION 0.01

#define SPH_Vs2 3e6
//#define SPH_Vs 2e3
#define SPH_c_ab 3.0e3
#define K_SPH_PD 1.0e-6
#define K_PD_PD 1.0e-6
#define SPH_SPH_mu 5e-4
#define SPH_BP_mu 1e-4


#define PERIDYNAMICS_MIN_DISTANCE 0.0005
#define PD_V (0.0005*0.0005*0.0005)

#define PD_K ((5.0)*(2.94315e9))
#define Kc (1.0e7)

//#define s0 0.003
//#define s0dev 0.0001

#define LWS 192
#define VDIM 128.0

#define FCELL (0.75)
#define MAXCLIST 128

#define MAX_PD_BOND_COUNT 528
//#define PHASEIN (3000) // for peridynamics
#define PHASEIN (30)

//#define PD_MATRIX_DIM 3
//#define PD_MATRIX_BLOCK_DIM 9


#define alpha_pr 0.25

#define CELL_NO_PARTICLE 0
#define CELL_SPH 1
#define CELL_PD 2
#define CELL_SPH_PD 3
#define CELL_HAS_NEIGHBOR_SPH_PD 4
#define CELL_SEMI_ACTIVE 5


#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

//enum Preconditions
//{
//    IDENTITY = 0,
//    INCOMPLETE_CHOLESKY
//};

//#define PRECONDITIONER (0)
#define MAX_CG_ITERATION (100000)
#define CG_RELATIVE_TOLERANCE (1e-20)

struct Clist
{
    int plist[MAXCLIST];
    int plist_top;
    int dummy[7];
};

struct Mat3x3
{
    real3_t row[3];
};
#define SIZE_MAT_3X3 (sizeof(real3_t)*3)

//enum WorkingBond
//{
//    KEEP_BOND = 0x0,
//    REMOVE_BOND = 0x1
//};

enum Integrator
{
    IMPLICIT_EULER =0,
    NEWMARK_BETA
};

#endif // CONSTANTS_H

