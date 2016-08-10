//------------------------------------------------------------------------------------------
//
//
// Created on: 4/23/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef IMPLICIT_EULER_CUH
#define IMPLICIT_EULER_CUH

#include <cuda_runtime.h>

#include "cg_solver.cuh"
#include "parameters.h"
#include "definitions.h"

extern __constant__ SimulationParameters simParams;

//------------------------------------------------------------------------------------------
__global__ void fillMatrixImplicitEuler(int* pd_activity,
                                        int* pd_bond_list_top,
                                        Mat3x3* pd_system_matrix)
{
    int p = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    if(pd_activity[p] == INVISIBLE)
    {
        return;
    }

    int bond_top_index = pd_bond_list_top[p];
    real_t timestep2 = simParams.pd_time_step * SYSTEM_TIMESTEP_BASE;
    timestep2 = timestep2 * timestep2; // squared

    Mat3x3 mass_matrix;
    setDiagonalMatrix(mass_matrix, simParams.pd_particle_mass);
    Mat3x3 derivative_matrix = pd_system_matrix[p];

    matrixMultiplyNumber(derivative_matrix, timestep2);
    matrixPlus(derivative_matrix, mass_matrix);
    pd_system_matrix[p] = derivative_matrix;

    for (int bond = 0; bond <= bond_top_index; ++bond)
    {
        int element_index = (bond + 1) * simParams.num_pd_particle + p;
        derivative_matrix = pd_system_matrix[element_index];
        matrixMultiplyNumber(derivative_matrix, timestep2);
        pd_system_matrix[element_index] = derivative_matrix;

    }

}

//------------------------------------------------------------------------------------------
__global__ void fillVectorBImplicitEuler(int* pd_activity,
                                         real4_t* pd_force,
                                         real4_t* pd_velocity,
                                         real4_t* pd_system_vector)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_pd_particle)
    {
        return;
    }

    if(pd_activity[index] == INVISIBLE)
    {
        return;
    }

    real_t timestep = simParams.pd_time_step * SYSTEM_TIMESTEP_BASE;
    pd_system_vector[index] = simParams.pd_particle_mass * pd_velocity[index] +
                              timestep * pd_force[index];

}


//------------------------------------------------------------------------------------------
__global__ void updatePDPositionImplicitEuler(int* _pdActivity,
                                              real4_t* _position,
                                              real4_t* _velocity)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_pd_particle)
    {
        return;
    }

    if(_pdActivity[index] == INVISIBLE)
    {
        return;
    }

    real4_t newVel = _velocity[index];
    real4_t oldPos =  _position[index];


    // correct the position
    real_t radius = simParams.pd_particle_radius;
    real_t min_x = simParams.boundary_min_x + radius;
    real_t min_y = simParams.boundary_min_y + radius;
    real_t min_z = simParams.boundary_min_z + radius;

    real_t max_x = simParams.boundary_max_x - radius;
    real_t max_y = simParams.boundary_max_y - radius;
    real_t max_z = simParams.boundary_max_z - radius;

    real4_t newPos =  oldPos + newVel * SYSTEM_TIMESTEP_BASE * simParams.pd_time_step;

    if (newPos.x > max_x)
    {
        newVel.x *= -RESTITUTION;

        newPos.x = max_x - RESTITUTION * (newPos.x - max_x);
    }

    if (newPos.x < min_x)
    {
        newVel.x *= -RESTITUTION;

        newPos.x = min_x + RESTITUTION * (min_x - newPos.x);
    }

    if (newPos.y > max_y)
    {
        newVel.y *= -RESTITUTION;

        newPos.y = max_y - RESTITUTION * (newPos.y - max_y);
    }

    if (newPos.y < min_y)
    {
        newVel.y *= -RESTITUTION;

        newPos.y = min_y + RESTITUTION * (min_y - newPos.y);
    }

    if (newPos.z > max_z)
    {
        newVel.z *= -RESTITUTION;

        newPos.z = max_z - RESTITUTION * (newPos.z - max_z);
    }

    if (newPos.z < min_z)
    {
        newVel.z *= -RESTITUTION;

        newPos.z = min_z + RESTITUTION * (min_z - newPos.z);
    }


    __syncthreads();

    _position[index] = newPos;
    _velocity[index] = newVel;

}




#endif // IMPLICIT_EULER_CUH
