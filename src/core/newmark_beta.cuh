//------------------------------------------------------------------------------------------
//
//
// Created on: 4/23/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef NEWMARK_BETA_CUH
#define NEWMARK_BETA_CUH

#include <cuda_runtime.h>

#include "cg_solver.cuh"
#include "parameters.h"
#include "definitions.h"

extern __constant__ SimulationParameters simParams;

//------------------------------------------------------------------------------------------
__global__ void fillMatrixNewmarkBeta(int* pd_activity,
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
    timestep2 = (timestep2 * timestep2) / 4.0; // h^2 / 4

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
__global__ void fillVectorBNewmarkBeta(int* pd_activity,
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
    pd_system_vector[index] = timestep * (simParams.pd_particle_mass * pd_velocity[index] +
                                          0.5 * timestep * pd_force[index]);

}

//------------------------------------------------------------------------------------------
__global__ void updatePDVelocityNewmarkBeta(int* pd_activity,
                                            real4_t* pd_velocity,
                                            real4_t* pd_delta_x)
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

    pd_velocity[index] = 2.0 / timestep * pd_delta_x[index] - pd_velocity[index];
}

//------------------------------------------------------------------------------------------
__global__ void updatePDPositionNewmarkBeta(int* pd_activity,
                                            real4_t* pd_position,
                                            real4_t* pd_velocity,
                                            real4_t* pd_delta_x)
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

    real4_t pvel = pd_velocity[index];
    real4_t old_pos =  pd_position[index];
    real4_t new_pos =  old_pos + pd_delta_x[index];


    // correct the position
    real_t radius = simParams.pd_particle_radius;
    real_t min_x = simParams.boundary_min_x + radius;
    real_t min_y = simParams.boundary_min_y + radius;
    real_t min_z = simParams.boundary_min_z + radius;

    real_t max_x = simParams.boundary_max_x - radius;
    real_t max_y = simParams.boundary_max_y - radius;
    real_t max_z = simParams.boundary_max_z - radius;


    if (new_pos.x > max_x)
    {
        pvel.x *= -RESTITUTION;

        new_pos.x = max_x - RESTITUTION * (new_pos.x - max_x);
    }

    if (new_pos.x < min_x)
    {
        pvel.x *= -RESTITUTION;

        new_pos.x = min_x + RESTITUTION * (min_x - new_pos.x);
    }

    if (new_pos.y > max_y)
    {
        pvel.y *= -RESTITUTION;

        new_pos.y = max_y - RESTITUTION * (new_pos.y - max_y);
    }

    if (new_pos.y < min_y)
    {
        pvel.y *= -RESTITUTION;

        new_pos.y = min_y + RESTITUTION * (min_y - new_pos.y);
    }

    if (new_pos.z > max_z)
    {
        pvel.z *= -RESTITUTION;

        new_pos.z = max_z - RESTITUTION * (new_pos.z - max_z);
    }

    if (new_pos.z < min_z)
    {
        pvel.z *= -RESTITUTION;

        new_pos.z = min_z + RESTITUTION * (min_z - new_pos.z);
    }


    __syncthreads();

    pd_position[index] = new_pos;
    pd_velocity[index] = pvel;

}



#endif // NEWMARK_BETA_CUH
