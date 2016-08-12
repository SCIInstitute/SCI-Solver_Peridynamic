//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <stdio.h>
#include <math.h>
#include "helper_math.h"
#include <math_constants.h>
#include <thrust/functional.h>

#include "definitions.h"
#include "parameters.h"
#include "cutil_math_ext.h"

//------------------------------------------------------------------------------------------
// simulation parameters in constant memory:
__constant__ SimulationParameters simParams;

//------------------------------------------------------------------------------------------
// MISC functions
//------------------------------------------------------------------------------------------

// this kernel need to multiply with kernel_poly6_coeff
__host__ __device__ __forceinline__ real_t kernel_poly6(const real_t x)
{
    return (x >= 1) ? 0 : ((1.0 - x) * (1.0 - x) * (1.0 - x));
}

__host__ __device__ __forceinline__ real_t derivative_kernel_spiky(const real_t r)
{
    const real_t tmp = 1.0 - r / simParams.sph_kernel_smooth_length;
    return -(tmp * tmp) * simParams.sph_kernel_spiky;
}

__host__ __device__ __forceinline__ real_t derivative_kernel_3(const real_t r)
{
    const real_t tmp = 1 - r / simParams.sph_kernel_smooth_length;

    return -(tmp * tmp * tmp) * simParams.sph_kernel_spiky * 4e-4;
}

//__host__ __device__ __forceinline__ real_t derivative_kernel_2(const real_t r)
//{
//    const real_t tmp = 1 - r / simParams.sph_kernel_smooth_length;

//    return -(tmp * tmp) * simParams.sph_kernel_spiky * 1e-3;
//}

__host__ __device__ __forceinline__ real_t kernel_laplace(const real_t r)
{
    return (1.0 - r / simParams.sph_kernel_smooth_length) * simParams.sph_kernel_spiky * 2e9;
}

__device__ __forceinline__ int3 calcGridPos(real4_t p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - simParams.boundary_min_x) / simParams.cell_size);
    gridPos.y = floor((p.y - simParams.boundary_min_y) / simParams.cell_size);
    gridPos.z = floor((p.z - simParams.boundary_min_z) / simParams.cell_size);
    return gridPos;
}



__device__ __forceinline__ int calcGridHash(int3 gridPos)
{
    if(gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 ||
            gridPos.x >= simParams.num_cell_x ||
            gridPos.y >= simParams.num_cell_y ||
            gridPos.z >= simParams.num_cell_z)
    {
        return -1000;
    }

//    gridPos.x = (gridPos.x < simParams.num_cell_x) ? gridPos.x : simParams.num_cell_x - 1;
//    gridPos.y = (gridPos.y < simParams.num_cell_y) ? gridPos.y : simParams.num_cell_y - 1;
//    gridPos.z = (gridPos.z < simParams.num_cell_z) ? gridPos.z : simParams.num_cell_z - 1;

    return __umul24(__umul24(gridPos.z, simParams.num_cell_y), simParams.num_cell_x)
           + __umul24(gridPos.y, simParams.num_cell_x) + gridPos.x;
}

__device__ __forceinline__ bool isBoundaryCell(int3& gridPos)
{
    if((gridPos.x == 0) || (gridPos.x == simParams.num_cell_x - 1) ||
            (gridPos.y == 0) || (gridPos.y == simParams.num_cell_y - 1) ||
            (gridPos.z == 0) || (gridPos.z == simParams.num_cell_z - 1))
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ __forceinline__ real_t densityBP(int startIndex, int endIndex,
        real4_t pos_i2, real4_t* _BPPosition)
{
    real4_t diff_pos;
    real_t dist_sq;
    real_t tmp = 0.0;

    for (int j = startIndex; j < endIndex; ++j)
    {
        diff_pos = pos_i2 - _BPPosition[j];
        dist_sq = diff_pos.x * diff_pos.x + diff_pos.y * diff_pos.y
                  + diff_pos.z * diff_pos.z;

        tmp += kernel_poly6(dist_sq / simParams.sph_kernel_smooth_length_squared);

    }

    return tmp;
}




__host__ __device__ __forceinline__ bool isEqual(real_t x, real_t y)
{
    return (fabs(x - y) < 1e-9);
}

//------------------------------------------------------------------------------------------
__global__ void initSPHParticlesData(int* _activity,
                                     int* _validity,
                                     int* _sphTimestep,
                                     real4_t* _velocity,
                                     real4_t* _force,
                                     real_t* _sortedDensity,
                                     real_t* _sortedNormalizedDensity,
                                     real_t _initial_sph_velocity)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

//    _activity[index] = INVISIBLE;
    _activity[index] = ACTIVE;
    _validity[index] = 0;
    _sphTimestep[index] = TIMESTEP_SPH;
    _velocity[index] = MAKE_REAL4(0.0, 0.0, _initial_sph_velocity, 0.0);
    _force[index] = MAKE_REAL4(0.0, 0.0, 0.0, 0.0);
    _sortedDensity[index] = simParams.sph_rest_density;
    _sortedNormalizedDensity[index] = simParams.sph_rest_density;

}

//------------------------------------------------------------------------------------------
__global__ void findActiveSPHParticles(int* _activity,
                                       int* _validity)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    if(_activity[index] == INVISIBLE)
    {
        return;
    }

    _activity[index] = (_validity[index] <= 0) ? ACTIVE : INACTIVE;
}


//------------------------------------------------------------------------------------------
__global__ void timestepLimiterCell(int* _cellParticleType,
                                    int* _sphActivity,
                                    int* _sphTimestep,
                                    int* _sphParticleUnsortedIndex,
                                    int* _sphCellStartIndex,
                                    int* _sphCellEndIndex)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_cells)
    {
        return;
    }

    // 3: cell has 2 types of particle
    // 4: cell has 1 type of particle, but its neighbor has 2
    if(_cellParticleType[index] == CELL_SPH_PD ||
            _cellParticleType[index] == CELL_HAS_NEIGHBOR_SPH_PD)
    {

        int startIndex = FETCH(_sphCellStartIndex, index);
//    printf("cell: %d, start = %d , ty=%d\n" , index, startIndex, _cellParticleType[index]);

        if (startIndex != 0xffffffff)       // cell is not empty
        {
            int endIndex = FETCH(_sphCellEndIndex, index);

            for(int j = startIndex; j < endIndex; ++j)
            {
                int pIndex = _sphParticleUnsortedIndex[j];

                _sphActivity[pIndex] = ACTIVE;
                _sphTimestep[pIndex] = TIMESTEP_PD;
            }


        } // if
    }


    if(_cellParticleType[index] == CELL_SEMI_ACTIVE)
    {

        int startIndex = FETCH(_sphCellStartIndex, index);
//    printf("cell: %d, start = %d , ty=%d\n" , index, startIndex, _cellParticleType[index]);

        if (startIndex != 0xffffffff)       // cell is not empty
        {
            int endIndex = FETCH(_sphCellEndIndex, index);

            for(int j = startIndex; j < endIndex; ++j)
            {
                int pIndex = _sphParticleUnsortedIndex[j];

                if(_sphActivity[pIndex] == INACTIVE)
                {
                    _sphActivity[pIndex] = SEMI_ACTIVE;
                }
            }


        } // if
    }

}

//------------------------------------------------------------------------------------------
__global__ void updateSPHVelocity(int* _activity,
                                  int* _timestep,
                                  real4_t* _velocity,
                                  real4_t* _force)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    if(_activity[index] != ACTIVE)
    {
        return;
    }

//    REAL mass = isSPH(index) ? simParams.sph_particle_mass : simParams.pd_particle_mass;
    real_t timestep = SYSTEM_TIMESTEP_BASE * _timestep[index];

    _velocity[index] += _force[index] * timestep / 2.0; // / mass;

}

//------------------------------------------------------------------------------------------
__global__ void updatePDVelocity(int* _pdActivity,
                                 real4_t* _velocity,
                                 real4_t* _deltaV)
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

    _velocity[index] += _deltaV[index];

}

////------------------------------------------------------------------------------------------
//__global__ void updatePDVelocity(int* _pdActivity,
//                                 real4_t* _velocity,
//                                 real4_t* _force)
//{
//    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

//    if (index >= simParams.num_pd_particle)
//    {
//        return;
//    }

//    if(_pdActivity[index] == INVISIBLE)
//    {
//        return;
//    }

//    _velocity[index] += _force[index] * SYSTEM_TIMESTEP_BASE / 2.0; // / mass;

//}

//------------------------------------------------------------------------------------------
__global__ void updateSPHPosition(int* _activity,
                                  int* _sphValidity,
                                  int* _timestep,
                                  real4_t* _position,
                                  real4_t* _velocity)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    int pActivity = _activity[index];

    if(pActivity == INACTIVE ||
            pActivity == SEMI_ACTIVE)
    {
        return;
    }

    if(pActivity == INVISIBLE && _sphValidity[index] > 0)
    {
        return;
    }

    real_t timestep = SYSTEM_TIMESTEP_BASE * _timestep[index];

    real4_t newVel = _velocity[index];
    real4_t oldPos =  _position[index];
    real4_t newPos =  oldPos + newVel * timestep;


    // correct the position
    real_t radius = simParams.sph_particle_radius;
    real_t min_x = simParams.boundary_min_x + radius;
    real_t min_y = simParams.boundary_min_y + radius;
    real_t min_z = simParams.boundary_min_z + radius;

    real_t max_x = simParams.boundary_max_x - radius;
    real_t max_y = simParams.boundary_max_y - radius;
    real_t max_z = simParams.boundary_max_z - radius;



    if(pActivity == INVISIBLE)
    {
        if(newPos.x > min_x + simParams.sph_kernel_smooth_length &&
                newPos.x < max_x - simParams.sph_kernel_smooth_length &&
                newPos.y > min_y + simParams.sph_kernel_smooth_length &&
                newPos.y < max_y - simParams.sph_kernel_smooth_length &&
                newPos.z > min_z + simParams.sph_kernel_smooth_length &&
                newPos.z < max_z - simParams.sph_kernel_smooth_length)
        {
            _activity[index] = ACTIVE;
        }

    }
    else
    {


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

        _velocity[index] = newVel;
    }

//    __syncthreads();

    _position[index] = newPos;


}

//------------------------------------------------------------------------------------------
__global__ void findCellParticleType(int* _cellParticleType,
                                     int* _sphCellStartIndex,
                                     int* _pdCellStartIndex,
                                     int* _sphCellEndIndex,
                                     int* _pdCellEndIndex)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_cells)
    {
        return;
    }


    int hasSPH = CELL_NO_PARTICLE;
    int hasPD = CELL_NO_PARTICLE;

    int sphStartIndex = FETCH(_sphCellStartIndex, index);
    int pdStartIndex = FETCH(_pdCellStartIndex, index);

    if (sphStartIndex != 0xffffffff)
    {
        hasSPH = CELL_SPH;
    }

    if (pdStartIndex != 0xffffffff)
    {
        hasPD = CELL_PD;
    }

    _cellParticleType[index] = hasSPH + hasPD;

//    if(hasSPH + hasPD == CELL_SPH_PD)
//    {
//        printf("cell %d has SPH_PD, start- %d, %d, end: %d, %d , %d\n", index, sphStartIndex,
//               pdStartIndex,
//               FETCH(_sphCellEndIndex, index),  FETCH(_pdCellEndIndex, index), 0xffffffff);
//    }

}

//------------------------------------------------------------------------------------------
__global__ void propagateMixedCell(int* _cellParticleType)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_cells)
    {
        return;
    }

    int thisCellType = _cellParticleType[index];

//    if(thisCellType == CELL_SPH_PD ||
//            thisCellType == CELL_NO_PARTICLE)
    if(thisCellType != CELL_SPH)
        // cell that has two types of particle: don't need to do anything
    {
        return;
    }

    int neighborIndex;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neighborIndex = (int) index + x + y * simParams.num_cell_x + z * simParams.num_cell_x
                                * simParams.num_cell_y;

                if(neighborIndex < 0 || neighborIndex >= simParams.num_cells)
                {
                    continue;
                }

//                printf("cell %d, neigh = %d\n", index, neighborIndex);

//                if(thisCellType == CELL_SPH)
//                {
                if(_cellParticleType[neighborIndex] == CELL_SPH_PD ||
                        _cellParticleType[neighborIndex] == CELL_PD)
                {
                    _cellParticleType[index] = CELL_HAS_NEIGHBOR_SPH_PD;
                    return;
                }

//                }
//                else
//                {
//                    if(_cellParticleType[neighborIndex] == CELL_SPH_PD ||
//                            _cellParticleType[neighborIndex] == CELL_SPH)
//                    {
//                        _cellParticleType[index] = CELL_HAS_NEIGHBOR_SPH_PD;
//                        return;
//                    }
//                }

            } // for x
        } // for y
    } // for z


}


//------------------------------------------------------------------------------------------
__global__ void findSemiActiveCell(int* _cellParticleType)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_cells)
    {
        return;
    }

    if(_cellParticleType[index] != CELL_SPH)
        // cell not only has SPH: don't need to do anything
    {
        return;
    }

    int neighborIndex;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neighborIndex = (int)index + x + y * simParams.num_cell_x + z * simParams.num_cell_x
                                * simParams.num_cell_y;

                if(neighborIndex < 0 || neighborIndex >= simParams.num_cells)
                {
                    continue;
                }

                if(_cellParticleType[neighborIndex] == CELL_HAS_NEIGHBOR_SPH_PD)
                {
                    _cellParticleType[index] = CELL_SEMI_ACTIVE; // cell type = 5: semi active
                    return;
                }

            } // for x
        } // for y
    } // for z


}


//------------------------------------------------------------------------------------------
__global__ void updateSPHValidity(int* _validity, int _numStep)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    //    _validity[index] -= _numMovingStep[0];
    _validity[index] -= _numStep; // assume that numMovingStep = 1

}


//------------------------------------------------------------------------------------------

// From Markus Gross:
#define PRIME1 73856093
#define PRIME2 19349663
#define PRIME3 83492791

__device__ __forceinline__ int lookupcl(const real4_t& pos)
{
    long int ix, iy, iz;
    int hash;

    ix = (long int) (pos.x / (FCELL * PERIDYNAMICS_MIN_DISTANCE)) + (long int) 128;
    iy = (long int) (pos.y / (FCELL * PERIDYNAMICS_MIN_DISTANCE)) + (long int) 128;
    iz = (long int) (pos.z / (FCELL * PERIDYNAMICS_MIN_DISTANCE)) + (long int) 128;

    if (ix < 0)
    {
        return (-1);
    }

    if (iy < 0)
    {
        return (-1);
    }

    if (iz < 0)
    {
        return (-1);
    }

    hash = (int) (((ix * PRIME1) ^ (iy * PRIME2) ^ (iz * PRIME3))
                  % simParams.num_clists);
    return (hash);
}



__global__ void collectPDParticles(int* _pdActivity,
                                   real4_t* _pdPosition,
                                   struct Clist* _pdNeighborList)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int wci, plist_slot;

    if(index >= simParams.num_pd_particle)
    {
        return;
    }

    if(_pdActivity[index] == INVISIBLE)
    {
        return;
    }

    wci = lookupcl(_pdPosition[index]);

    if(wci == -1)
    {
        return;
    }

    plist_slot = atomicAdd(&_pdNeighborList[wci].plist_top, 1);

// The plist_top starts at -1, denoting empty list, and atom_inc returns
// the old value, before the increment; we want the new one:
    plist_slot++;

    if(plist_slot >= MAXCLIST)
    {
        // plist_top is the last legal slot marker; we can't let it grow.
        // If the hashing function is good, this doesn't happen.
        atomicSub(&_pdNeighborList[wci].plist_top, 1);
        return;
    }

    _pdNeighborList[wci].plist[plist_slot] = index;
}

//------------------------------------------------------------------------------------------
__global__ void collidePDParticles(int* _pdActivity,
                                   real4_t* _pdForce,
                                   const real4_t* _pdPosition,
                                   struct Clist* __restrict__ _pdNeighborList)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int walker, q, wci;
    real_t r, howfar, mag;
    int idx, idy, idz;
    real4_t next, diff;
    real4_t ppos = _pdPosition[index];
    real4_t force = MAKE_REAL4_FROM_REAL(0);

    if(index >= simParams.num_pd_particle)
    {
        return;
    }

    if(_pdActivity[index] == INVISIBLE)
    {
        return;
    }

// This is restricted to collision effects.  The compression effect
// among those bonded to one another is handled by bondforces.

    for(idx = -1; idx <= 1; ++idx)
    {
        for(idy = -1; idy <= 1; ++idy)
        {
            for(idz = -1; idz <= 1; ++idz)
            {
                diff = MAKE_REAL4(idx * PERIDYNAMICS_MIN_DISTANCE, idy * PERIDYNAMICS_MIN_DISTANCE,
                                  idz * PERIDYNAMICS_MIN_DISTANCE, 0.0);
                next = ppos + FCELL * diff;

                if((wci = lookupcl(next)) != -1)
                {
                    for(walker = _pdNeighborList[wci].plist_top; walker >= 0;
                            walker--)
                    {
                        q = _pdNeighborList[wci].plist[walker];

                        if(q == index)
                        {
                            continue;
                        }

                        real4_t pq = ppos - _pdPosition[q];
//                        r = length(pq);
                        r = sqrt( pq.x * pq.x + pq.y * pq.y + pq.z * pq.z);

                        if(r >= FCELL * PERIDYNAMICS_MIN_DISTANCE)
                        {
                            continue;
                        }

                        if(r == 0.0)
                        {
                            continue;
                        }

                        howfar = (r - FCELL * PERIDYNAMICS_MIN_DISTANCE);
                        mag = Kc * (howfar * howfar);
                        force += (mag / PD_V) * (pq / r);
//                        force += mag * (pq / r);
//                        real4_t ff = mag * (pq / r);
//                        printf("collide wci = %d, %d-%d: %f, %f, %f,   %f, %f, %f, dist=%f/%f, scale=%f, "
//                               "force = %f, %f, %f\n", wci,
//                               index,
//                               q, simParams.scaleFactor * ppos.x, simParams.scaleFactor * ppos.y,
//                               simParams.scaleFactor * ppos.z,
//                               simParams.scaleFactor * _pdPosition[q].x, simParams.scaleFactor * _pdPosition[q].y,
//                               simParams.scaleFactor * _pdPosition[q].z,
//                               r, FCELL * PERIDYNAMICS_MIN_DISTANCE,
//                               simParams.scaleFactor ,
//                               ff.x, ff.y, ff.z);
                    }
                }
            }
        }
    }

    if(length(force) > 1e-10)
    {
        _pdForce[index] += force;
    }

//    printf("%d: nei %d\n", index, _pdNeighborList[lookupcl(ppos)].plist_top );
}
//------------------------------------------------------------------------------------------
__global__ void correctCollidedPDParticle(int* _pdActivity,
        real4_t* _pdVelocity,
        const real4_t* _pdPosition,
        struct Clist* __restrict__ _pdNeighborList)
{

    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if(index >= simParams.num_pd_particle)
    {
        return;
    }

    int walker, q, wci;
    int idx, idy, idz;
    real4_t next, diff;
    real4_t ppos = _pdPosition[index];

    if(_pdActivity[index] == INVISIBLE)
    {
        return;
    }

    real_t minDist = 1e100;
    int qMinDist;

    for(idx = -1; idx <= 1; ++idx)
    {
        for(idy = -1; idy <= 1; ++idy)
        {
            for(idz = -1; idz <= 1; ++idz)
            {
                diff = MAKE_REAL4(idx * PERIDYNAMICS_MIN_DISTANCE, idy * PERIDYNAMICS_MIN_DISTANCE,
                                  idz * PERIDYNAMICS_MIN_DISTANCE, 0.0);
                next = ppos + FCELL * diff;

                if((wci = lookupcl(next)) != -1)
                {
                    for(walker = _pdNeighborList[wci].plist_top; walker >= 0;
                            walker--)
                    {
                        q = _pdNeighborList[wci].plist[walker];

                        if(q == index)
                        {
                            continue;
                        }

                        real4_t pq = ppos - _pdPosition[q];
                        real_t r = sqrt( pq.x * pq.x + pq.y * pq.y + pq.z * pq.z);

                        if(r < FCELL * PERIDYNAMICS_MIN_DISTANCE && r > 0.0)
                        {
                            minDist = r;
                            qMinDist = q;
                        }

                    }
                }
            }
        }
    }


    if(minDist < FCELL * PERIDYNAMICS_MIN_DISTANCE)
    {

        real4_t vin, vit, vjn, vjt;
        real4_t vin2, vjn2;

        real4_t eij = (ppos - _pdPosition[qMinDist]) / minDist;

        //                        correctedPos = posq- simParams.pd_particle_radius* eij;

        real4_t velp = _pdVelocity[index];
        vin = eij * (eij.x * velp.x + eij.y * velp.y + eij.z * velp.z);
        vit = velp - vin;

        real4_t velq = _pdVelocity[q];
        vjn = eij * (eij.x * velq.x + eij.y * velq.y + eij.z * velq.z);
        vjt = velq - vjn;

        vin2 = (vin + vjn) / 2.0;
        vjn2 = vin2;

        _pdVelocity[index] = vin2 + vit;

        _pdVelocity[qMinDist] = vjn2 + vjt;
    }
}

//------------------------------------------------------------------------------------------
__global__ void calculateCellHash(int* _particleCellHash,
                                  int* _particleUnsortedIndex,
                                  real4_t* _position,
                                  int _numParticles)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= _numParticles)
    {
        return;
    }

    volatile real4_t ppos = _position[index];

    // get address in grid
    int3 gridPos = calcGridPos(ppos);
    int hash = calcGridHash(gridPos);


    __syncthreads();
    // store grid hash and particle index
    _particleCellHash[index] = hash;
    _particleUnsortedIndex[index] = index;


}


//------------------------------------------------------------------------------------------
__global__ void initSPHParticleTimestep(int* _validity,
                                        int* _timestep)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;


    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    if(_validity[index] > 0)
    {
        return;
    }

    _timestep[index] = TIMESTEP_SPH;
    _validity[index] = TIMESTEP_SPH;
}

//------------------------------------------------------------------------------------------
__global__ void collectSPHParticlesToCells(int* _cellStartIndex,
        int* _cellEndIndex,
        int* _particleCellHash,
        int* _particleUnsortedIndex,
        real4_t* _position,
        real4_t* _velocity,
        real4_t* _sortedPos,
        real4_t* _sortedVel,
        int _numParticles)
{
    extern __shared__ int sharedHash[];    // blockSize + 1 elements
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= _numParticles)
    {
        return;
    }

    int hash = _particleCellHash[index];


//    printf("%d: %d\n", index, hash);
    // Load hash data into shared memory so that we can look
    // at neighboring particle's hash value without loading
    // two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;


    if (index > 0 && threadIdx.x == 0)
    {
        // first thread in block must load neighbor particle hash
        sharedHash[0] = _particleCellHash[index - 1];
    }

    __syncthreads();

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if(hash >= 0)
    {
        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            _cellStartIndex[hash] = index;

            if (index > 0 && sharedHash[threadIdx.x] >= 0)
            {
                _cellEndIndex[sharedHash[threadIdx.x]] = index;
            }
        }

        if (index == _numParticles - 1)
        {
            _cellEndIndex[hash] = index + 1;
        }
    }


    // Now use the sorted index to reorder the pos and vel data
    int unsortedIndex = _particleUnsortedIndex[index];
    real4_t pos = FETCH(_position,
                        unsortedIndex); // macro does either global read or texture fetch
    real4_t vel = FETCH(_velocity, unsortedIndex);     // see particles_kernel.cuh

    _sortedPos[index] = pos;
    _sortedVel[index] = vel;


//    printf("%d:: sortedpos = %e, %e, %e\n", index,  pos.x, pos.y, pos.z);

}


//------------------------------------------------------------------------------------------
__global__ void collectPDParticlesToCells(int* _cellStartIndex,
        int* _cellEndIndex,
        int* _particleCellHash,
        int* _particleUnsortedIndex,
        real4_t* _position,
        real4_t* _velocity,
        real4_t* _sortedPos,
        real4_t* _sortedVel,
        int _numParticles)
{
    extern __shared__ int sharedHash[];    // blockSize + 1 elements
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= _numParticles)
    {
        return;
    }

    int hash = _particleCellHash[index];


//    printf("%d: %d\n", index, hash);
    // Load hash data into shared memory so that we can look
    // at neighboring particle's hash value without loading
    // two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;


    if (index > 0 && threadIdx.x == 0)
    {
        // first thread in block must load neighbor particle hash
        sharedHash[0] = _particleCellHash[index - 1];
    }

    __syncthreads();

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if(hash >= 0)
    {
        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            _cellStartIndex[hash] = index;

            if (index > 0 && sharedHash[threadIdx.x] >= 0)
            {
                _cellEndIndex[sharedHash[threadIdx.x]] = index;
            }
        }

        if (index == _numParticles - 1)
        {
            _cellEndIndex[hash] = index + 1;
        }
    }


    // Now use the sorted index to reorder the pos and vel data
    int unsortedIndex = _particleUnsortedIndex[index];
    real4_t pos = FETCH(_position,
                        unsortedIndex); // macro does either global read or texture fetch
    real4_t vel = FETCH(_velocity, unsortedIndex);     // see particles_kernel.cuh

    _sortedPos[index] = pos;
    _sortedVel[index] = vel;


//    printf("%d:: sortedpos = %e, %e, %e\n", index,  pos.x, pos.y, pos.z);

}

//------------------------------------------------------------------------------------------
__global__ void collectPeridynamicsParticlesToCells(int* _pdCellStartIndex,
        int* _pdCellEndIndex,
        int* _pdParticleCellHash,
        int* _pdParticleUnsortedIndex,
        real4_t* _position,
        real4_t* _pdSortedPos)
{
    extern __shared__ int sharedHash[];    // blockSize + 1 elements
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_pd_particle)
    {
        return;
    }

    int hash = _pdParticleCellHash[index];

    // Load hash data into shared memory so that we can look
    // at neighboring particle's hash value without loading
    // two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;

    if (index > 0 && threadIdx.x == 0)
    {
        // first thread in block must load neighbor particle hash
        sharedHash[0] = _pdParticleCellHash[index - 1];
    }

    __syncthreads();

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell.
    // As it isn't the first particle, it must also be the cell end of
    // the previous particle's cell

    if (index == 0 || hash != sharedHash[threadIdx.x])
    {
        _pdCellStartIndex[hash] = index;

        if (index > 0)
        {
            _pdCellEndIndex[sharedHash[threadIdx.x]] = index;
        }
    }

    if (index == simParams.num_pd_particle - 1)
    {
        _pdCellEndIndex[hash] = index + 1;
    }

    // Now use the sorted index to reorder the pos and vel data
    int unsortedIndex = _pdParticleUnsortedIndex[index];
    real4_t pos = FETCH(_position,
                        unsortedIndex); // macro does either global read or texture fetch

    _pdSortedPos[index] = pos;

}


//------------------------------------------------------------------------------------------
__device__ __forceinline__  real4_t forces_SPH_SPH(real4_t _eij,
        real4_t _relVelocity,
        real_t _dist,
        real_t _pressureI,
        real_t _pressureJ,
        real_t _densityI,
        real_t _densityJ, int index)
{
    real4_t force = MAKE_REAL4_FROM_REAL(0.0);

    if (_dist >= simParams.sph_kernel_smooth_length || _dist == 0.0)
    {
        return force;
    }


    real_t fp = (_pressureI / _densityI / _densityI + _pressureJ / _densityJ / _densityJ)
                * _densityI; //simParams.sph_particle_mass * _densityI;
    real_t fv = simParams.sph_sph_viscosity / _densityJ; //* simParams.sph_particle_mass

    real_t pNear_i = SPH_Vs2 * _densityI / simParams.sph_rest_density;
    real_t pNear_j = SPH_Vs2 * _densityJ / simParams.sph_rest_density;

    real_t FcNear = (pNear_i / _densityI / _densityI
                     + pNear_j / _densityJ / _densityJ) * _densityI; //simParams.sph_particle_mass *


    force = fp * derivative_kernel_spiky(_dist) * _eij +
            fv * kernel_laplace(_dist) * _relVelocity +
            FcNear * derivative_kernel_3(_dist) * _eij;


    return force;
}


//------------------------------------------------------------------------------------------
__device__ __forceinline__ real4_t forcesSPHWithBoundary(real4_t _eij,
        real_t _dist,
        real4_t _velocityI,
        real_t _pressureI,
        real_t _densityI)
{
    real4_t force = MAKE_REAL4_FROM_REAL(0.0);

    if (_dist >= simParams.sph_kernel_smooth_length || _dist == 0.0)
    {
        return force;
    }

    real4_t relVel = -2.0 * _velocityI;

    real_t fp = (_pressureI / _densityI / _densityI) *
                _densityI; //simParams.sph_particle_mass *

    real_t fv = simParams.sph_boundary_viscosity /
                simParams.sph_rest_density; //* simParams.sph_particle_mass /


    real_t pNear_i = SPH_Vs2 * _densityI / simParams.sph_rest_density;
    real_t pNear_j = SPH_Vs2;

    real_t FcNear = 10.0 * (pNear_i / _densityI / _densityI
                            + pNear_j / simParams.sph_rest_density / simParams.sph_rest_density) * _densityI;
//                  simParams.sph_particle_mass *



    force = fp * derivative_kernel_spiky(_dist) * _eij  +
            fv * kernel_laplace(_dist) * relVel +
            FcNear * derivative_kernel_3(_dist) * _eij;

    return force;
}

__device__ __forceinline__ real4_t forceBP(real_t iDensity, real_t iPressure,
        real4_t vel_i, int startIndex, int endIndex,
        real4_t pos_i2, real4_t* _BPPosition)
{
    real4_t pForce = MAKE_REAL4_FROM_REAL(0.0);

    for (int j = startIndex; j < endIndex; ++j)
    {
        real4_t eij = _BPPosition[j] - pos_i2;
        real_t dist = sqrt(eij.x * eij.x + eij.y * eij.y + eij.z * eij.z);
        eij /= fmaxf(dist, (real_t) SMALL);

        pForce += forcesSPHWithBoundary(eij, dist, vel_i, iPressure, iDensity);
    }

    return pForce;
}

//------------------------------------------------------------------------------------------
__global__ void calculateSPHParticleForces(int* _activity,
        real4_t* _sortedPos,
        real4_t* _sortedVel,
        real4_t* _force,
        real_t* _sortedNormalizedDensity,
        real_t* _sortedPressure,
        int* _particleUnsortedIndex,
        int* _cellStartIndex,
        int* _cellEndIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_activity[pIndex] != ACTIVE)
    {
        return;
    }

    // read particle data from sorted arrays
    real4_t pos_i = FETCH(_sortedPos, index);
    real4_t vel_i = FETCH(_sortedVel, index);
    real4_t pForce = MAKE_REAL4_FROM_REAL(0.0);

    int3 cellPos = calcGridPos(pos_i);
    int neighborCellHash;
    int neighborCellStartIndex, neighborCellEndIndex;
    int neightborCellPos_x, neightborCellPos_y, neightborCellPos_z;
    real_t iDensity = _sortedNormalizedDensity[index];
    real_t iPressure = _sortedPressure[index];


//    printf("%d: or ind = %d, isSPH = %d, activity = %d\n", index, oi_index, isSPH(oi_index), _activity[oi_index]);

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neightborCellPos_x = (int) cellPos.x + x;
                neightborCellPos_y = (int) cellPos.y + y;
                neightborCellPos_z = (int) cellPos.z + z;

                if (neightborCellPos_x < 0 ||
                        neightborCellPos_y < 0 ||
                        neightborCellPos_z < 0 ||
                        neightborCellPos_x >= simParams.num_cell_x ||
                        neightborCellPos_y >= simParams.num_cell_y ||
                        neightborCellPos_z >= simParams.num_cell_z )
                {
                    continue;
                }

                neighborCellHash = calcGridHash(
                                       make_int3(neightborCellPos_x, neightborCellPos_y,
                                                 neightborCellPos_z));

                neighborCellStartIndex = FETCH(_cellStartIndex, neighborCellHash);

                if (neighborCellStartIndex != 0xffffffff) // cell is not empty
                {
                    // iterate over particles in this cell
                    neighborCellEndIndex = FETCH(_cellEndIndex, neighborCellHash);

                    for (int j = neighborCellStartIndex; j < neighborCellEndIndex;
                            ++j)
                    {
                        if(j == index)
                        {
                            continue;
                        }

                        real4_t eij = FETCH(_sortedPos, j) - pos_i;
                        real_t dist = sqrt(eij.x * eij.x + eij.y * eij.y + eij.z * eij.z);
                        eij /= fmaxf(dist, (real_t) SMALL);

                        real4_t relVel = FETCH(_sortedVel, j) - vel_i;
                        pForce += forces_SPH_SPH(eij, relVel, dist, iPressure, _sortedPressure[j],
                                                 iDensity, _sortedNormalizedDensity[j],
                                                 index);


                    }
                } // end startIndex...
            }
        }
    }


    // Gravity
    pForce.y -= (simParams.clockScale * 9.8); // simParams.sph_particle_mass * 9.8);


    _force[pIndex] = pForce;
}

//------------------------------------------------------------------------------------------
__global__ void calculateSPHParticleForcesBoundary(int* _activity,
        real4_t* _sortedPos,
        real4_t* _sortedVel,
        real4_t* _force,
        real_t* _sortedNormalizedDensity,
        real_t* _sortedPressure,
        real4_t* _BPPosition,
        int* _particleUnsortedIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;


    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_activity[pIndex] != ACTIVE)
    {
        return;
    }

    real4_t pos_i = FETCH(_sortedPos, index);
    real4_t vel_i = FETCH(_sortedVel, index);
    real_t iDensity = _sortedNormalizedDensity[index];
    real_t iPressure = _sortedPressure[index];
    int3 cellPos = calcGridPos(pos_i);

    if(!isBoundaryCell(cellPos))
    {
        return;
    }

    real4_t pForce = MAKE_REAL4_FROM_REAL(0.0);
    real4_t pos_i2;

    int startIndex = 0;
    int endIndex = 0;

    // bottom plane
    if (cellPos.y == 0)
    {
        startIndex = simParams.boundaryPlaneBottomIndex;
        endIndex = startIndex + simParams.boundaryPlaneBottomSize;


        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }


    // top plane
    if (cellPos.y == (simParams.num_cell_y - 1))
    {
        startIndex = simParams.boundaryPlaneTopIndex;
        endIndex = startIndex + simParams.boundaryPlaneTopSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }

// left side plane
    if (cellPos.x == 0)
    {
        startIndex = simParams.boundaryPlaneLeftSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneLeftSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }

// right side plane
    if (cellPos.x == (simParams.num_cell_x - 1))
    {
        startIndex = simParams.boundaryPlaneRightSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneRightSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }

// back plane
    if (cellPos.z == 0)
    {
        startIndex = simParams.boundaryPlaneBackIndex;
        endIndex = startIndex + simParams.boundaryPlaneBackSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }

// front plane
    if (cellPos.z == (simParams.num_cell_z - 1))
    {
        startIndex = simParams.boundaryPlaneFrontIndex;
        endIndex = startIndex + simParams.boundaryPlaneFrontSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        pForce += forceBP(iDensity, iPressure, vel_i, startIndex, endIndex, pos_i2, _BPPosition);
    }




    _force[pIndex] += pForce;
}

//------------------------------------------------------------------------------------------
__global__ void printCell(int* _cellParticleType)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_cells)
    {
        return;
    }

    if(_cellParticleType[index] == CELL_NO_PARTICLE)
    {
        return;
    }

    printf("Cell %d -- %d\n", index, _cellParticleType[index]);
}

//------------------------------------------------------------------------------------------
__global__ void collideDifferentParticles(int* _cellParticleType,
        real4_t* _sphPosition,
        real4_t* _sphSortedPos,
        real4_t* _sphSortedVel,
        real4_t* _sphVelocity,
        int* _sphParticleUnsortedIndex,
        real4_t* _pdSortedPos,
        real4_t* _pdSortedVel,
        real4_t* _pdVelocity,
        int* _pdParticleUnsortedIndex,
        int* _pdCellStartIndex,
        int* _pdCellEndIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    real4_t pos_i = FETCH(_sphSortedPos, index);

    int3 cellPos = calcGridPos(pos_i);
    int cellIndex = calcGridHash(cellPos);

    if(cellIndex < 0)
    {
        return;
    }

    if(_cellParticleType[cellIndex] != CELL_SPH_PD &&
            _cellParticleType[cellIndex] != CELL_HAS_NEIGHBOR_SPH_PD)
    {
        return;
    }

    int neightborCellPos_x, neightborCellPos_y, neightborCellPos_z;
    int neighborCellHash, startIndex, endIndex, j;


    real_t minDist = simParams.sph_pd_min_distance;
    real4_t minPosj;
    int minDistIndexj;
    real4_t eij;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neightborCellPos_x = (int) cellPos.x + x;
                neightborCellPos_y = (int) cellPos.y + y;
                neightborCellPos_z = (int) cellPos.z + z;

                if (neightborCellPos_x < 0 ||
                        neightborCellPos_y < 0 ||
                        neightborCellPos_z < 0 ||
                        neightborCellPos_x >= simParams.num_cell_x ||
                        neightborCellPos_y >= simParams.num_cell_y ||
                        neightborCellPos_z >= simParams.num_cell_z )
                {
                    continue;
                }

                neighborCellHash = calcGridHash(
                                       make_int3(neightborCellPos_x, neightborCellPos_y,
                                                 neightborCellPos_z));

                startIndex = FETCH(_pdCellStartIndex, neighborCellHash);

                if (startIndex != 0xffffffff)       // cell is not empty
                {
                    endIndex = FETCH(_pdCellEndIndex, neighborCellHash);

                    for (j = startIndex; j < endIndex; ++j)
                    {

                        real4_t pos_j = FETCH(_pdSortedPos, j);
                        eij = pos_j - pos_i;
                        real_t dist = sqrt(eij.x * eij.x + eij.y * eij.y + eij.z * eij.z);

                        if(dist < minDist)
                        {
                            minDist = dist;
                            minDistIndexj = j;
                            minPosj = pos_j;
                        }

                    }

                } // if

            } // for x
        } // for y
    } // for z


    if(minDist < simParams.sph_pd_min_distance)
    {
//        printf("%d: mindist = %f/ %f\n", index, minDist, simParams.sph_pd_min_distance);

        real4_t correctedPos = MAKE_REAL4_FROM_REAL(0);
        real4_t vin, vit, vjn, vjt;
        real4_t vin2, vit2, vjn2, vjt2;
        real_t mi = simParams.sph_particle_mass;
        real_t mj = simParams.pd_particle_mass;

        int sphIndex = _sphParticleUnsortedIndex[index];
        int pdIndex = _pdParticleUnsortedIndex[minDistIndexj];


        eij = minPosj - pos_i;
        eij /= fmaxf(minDist, (real_t) SMALL);

        correctedPos = minPosj - simParams.sph_pd_min_distance * eij;

        real4_t vel_i = _sphSortedVel[index];
        vin = eij * (eij.x * vel_i.x + eij.y * vel_i.y + eij.z * vel_i.z);
        vit = vel_i - vin;

        real4_t vel_j = _pdSortedVel[j];
        vjn = eij * (eij.x * vel_j.x + eij.y * vel_j.y + eij.z * vel_j.z);
        vjt = vel_j - vjn;

        vin2 = (mi * vin + mj * vjn) /
               (mi + mj);
        vjn2 = vin2;

        vit2 = ((mi + mj * simParams.sph_pd_slip) * vit +
                mj * (1.0 - simParams.sph_pd_slip) * vjt) / (mi + mj);
        vjt2 = ((mj + mi * simParams.sph_pd_slip) * vjt +
                mi * (1.0 - simParams.sph_pd_slip) * vit) / (mi + mj);

        _sphPosition[sphIndex] = correctedPos;
        _sphVelocity[sphIndex] = vin2 + vit2;

        vel_j = vjn2 + vjt2;
        _pdVelocity[pdIndex] = vel_j;

    }


}
//------------------------------------------------------------------------------------------
__global__ void calculatePeriParticlesForces(int* _pdActivity,
        real4_t* _position,
        real4_t* _force,
        real4_t* _PDOriginalPosition,
        real_t* _PDOriginalStretch,
        real_t* _PDStretch,
        real_t* _PDNewStretch,
        int* _PDBondList,
        int* _PDBondCount,
        real_t* _PDStiffness)
{
    int i_index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (i_index >= simParams.num_pd_particle)
    {
        return;
    }

    if(_pdActivity[i_index] == INVISIBLE)
    {
        return;
    }

    real4_t force = MAKE_REAL4_FROM_REAL(0.0);

    real4_t pos_i = FETCH(_position, i_index);
    real4_t opos_i = _PDOriginalPosition[i_index];

    real_t smin = 0.0;
    real_t r, vscale, d, drdivd, kk, ss0;
    int bondCount = _PDBondCount[i_index];
    real_t istretch = _PDStretch[i_index];


    for (int walker = 0; walker <= bondCount; ++walker)
    {
        int j_index = _PDBondList[walker * simParams.num_pd_particle + i_index];

        if(j_index < 0)
        {
            continue;
        }

        real4_t eij = FETCH(_position, j_index) - pos_i;
        r = sqrt(eij.x * eij.x + eij.y * eij.y + eij.z * eij.z);
        eij /= fmaxf(r, (real_t) SMALL);

//        relPos = FETCH(_position, j_index) - pos_i;
//        r = length(relPos);
//        r = fmaxf(r, (REAL) SMALL);

        vscale = 1.0;
        d = length(opos_i - _PDOriginalPosition[j_index]);


        if (d > (simParams.pd_horizon - simParams.pd_particle_radius))
        {
            vscale = 0.5 + (simParams.pd_horizon - d) / 2.0 / simParams.pd_particle_radius;
        }

        drdivd = r / d - 1.0;


        if (drdivd < smin)
        {
            smin = drdivd;
        }

        kk = drdivd * vscale;


        // max is used instead of min.  One compressive bond to particle p
        // increases its critical stetch for the next pass, and that increase
        // should make all bonds with p harder to break.
        if (drdivd > fmaxf(istretch, _PDStretch[j_index])) // && bondCount > 0)
        {
            _PDBondList[walker * simParams.num_pd_particle + i_index] = -1;
            continue;
        }


        force += kk * eij;

    }

    force *= simParams.pd_C_times_V * PD_K;// _PDStiffness[i_index];

    ss0 = (10.0 - 9.0 * simParams.clockScale) * _PDOriginalStretch[i_index];
    _PDNewStretch[i_index] = ss0 - alpha_pr * smin;

    force.y -= (simParams.clockScale * 9.8); // simParams.pd_particle_mass * 9.8);
    _force[i_index] = force;
}
//------------------------------------------------------------------------------------------
__global__ void calculateSPHParticleDensityBoundary(int* _activity,
        int* _cellParticleType,
        real_t* _sortedDensity,
        real4_t* _sortedPosition,
        real4_t* _BPPosition,
        int* _particleUnsortedIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }


    real4_t pos_i = FETCH(_sortedPosition, index);
    int3 cellPos = calcGridPos(pos_i);

    if(!isBoundaryCell(cellPos))
    {
        return;
    }

    int cellIndex = calcGridHash(cellPos);

    if(cellIndex < 0)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_cellParticleType[cellIndex] == CELL_SPH && _activity[pIndex] != ACTIVE)
    {
        return;
    }

    real_t tmp = 0.0;
    real4_t pos_i2;

    int startIndex = 0;
    int endIndex = 0;

    // bottom plane
    if (cellPos.y == 0)
    {
        startIndex = simParams.boundaryPlaneBottomIndex;
        endIndex = startIndex + simParams.boundaryPlaneBottomSize;


        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);

    }


    // top plane
    if (cellPos.y == (simParams.num_cell_y - 1))
    {
        startIndex = simParams.boundaryPlaneTopIndex;
        endIndex = startIndex + simParams.boundaryPlaneTopSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);

    }

// left side plane
    if (cellPos.x == 0)
    {
        startIndex = simParams.boundaryPlaneLeftSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneLeftSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// right side plane
    if (cellPos.x == (simParams.num_cell_x - 1))
    {
        startIndex = simParams.boundaryPlaneRightSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneRightSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// back plane
    if (cellPos.z == 0)
    {
        startIndex = simParams.boundaryPlaneBackIndex;
        endIndex = startIndex + simParams.boundaryPlaneBackSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// front plane
    if (cellPos.z == (simParams.num_cell_z - 1))
    {
        startIndex = simParams.boundaryPlaneFrontIndex;
        endIndex = startIndex + simParams.boundaryPlaneFrontSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }



    if(tmp > 0.0)
    {
        tmp *= (simParams.sph_kernel_poly6 * simParams.sph_particle_mass);

        // must be +=
        _sortedDensity[index] += tmp;
    }
}

//------------------------------------------------------------------------------------------
__global__ void initSPHParticleData(real4_t* sph_force,
                                    real_t* sph_density,
                                    real_t* sph_density_normalized,
                                    real_t* sph_pressure,
                                    int* sph_validity,
                                    int* sph_timestep)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    sph_force[index] = MAKE_REAL4_FROM_REAL(0.0);
    sph_density[index] = simParams.sph_rest_density;
    sph_density_normalized[index] = simParams.sph_rest_density;
    sph_pressure[index] = 0.0;

    sph_validity[index] = 0;
    sph_timestep[index] = TIMESTEP_SPH;
}

//------------------------------------------------------------------------------------------
__global__ void calculateSPHParticleDensity(int* _activity,
        int* _cellParticleType,
        real_t* _sortedDensity,
        real4_t* _sortedPosition,
        int* _particleUnsortedIndex,
        int* _cellStartIndex,
        int* _cellEndIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }


    real4_t pos_i = FETCH(_sortedPosition, index);

    int3 cellPos = calcGridPos(pos_i);
    int cellIndex = calcGridHash(cellPos);

    if(cellIndex < 0)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_cellParticleType[cellIndex] == CELL_SPH && _activity[pIndex] != ACTIVE)
    {
        return;
    }


//    printf("%d %d isNOT p density. Ce==%d, ac=%d\n", pIndex, simParams.num_pd_particle,
//           _cellParticleType[cellIndex] , _activity[pIndex] );



    real4_t diff_pos;
    real_t dist_sq;

    int neighborCellHash, startIndex, endIndex, j;
    int neightborCellPos_x, neightborCellPos_y, neightborCellPos_z;

// examine neighbouring cells
    real_t tmp = 0.0;

    // Loop over SPH particles
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neightborCellPos_x = (int) cellPos.x + x;
                neightborCellPos_y = (int) cellPos.y + y;
                neightborCellPos_z = (int) cellPos.z + z;

                if (neightborCellPos_x < 0 ||
                        neightborCellPos_y < 0 ||
                        neightborCellPos_z < 0 ||
                        neightborCellPos_x >= simParams.num_cell_x ||
                        neightborCellPos_y >= simParams.num_cell_y ||
                        neightborCellPos_z >= simParams.num_cell_z )
                {
                    continue;
                }

                neighborCellHash = calcGridHash(
                                       make_int3(neightborCellPos_x, neightborCellPos_y,
                                                 neightborCellPos_z));

                startIndex = FETCH(_cellStartIndex, neighborCellHash);

                if (startIndex != 0xffffffff)       // cell is not empty
                {
                    endIndex = FETCH(_cellEndIndex, neighborCellHash);

                    for (j = startIndex; j < endIndex; ++j)
                    {
                        diff_pos = pos_i - FETCH(_sortedPosition, j);
                        dist_sq = diff_pos.x * diff_pos.x
                                  + diff_pos.y * diff_pos.y
                                  + diff_pos.z * diff_pos.z;

                        tmp += kernel_poly6(dist_sq / simParams.sph_kernel_smooth_length_squared);

                    }
                }

            }
        }
    }

// Finally, multiply with the kernel coefficient
    tmp *= simParams.sph_kernel_poly6 * simParams.sph_particle_mass;

    _sortedDensity[index] = tmp;

//    if(tmp < 1e-8)
//    {
//        printf("%d: cellpos = %d, %d, %d, cellhash = %d, pos=%f, %f, %f\n", index, cellPos.x,
//               cellPos.y,
//               cellPos.z, cellIndex
//               , pos_i.x, pos_i.y, pos_i.z);
//    }

//    if(_particleUnsortedIndex[index] == 6206 || _particleUnsortedIndex[index] == 6207)
//    {
//        printf("unsorte: %d, ac=%d, den=%f\n", _particleUnsortedIndex[index],
//               _activity[_particleUnsortedIndex[index]], tmp);
//    }
}

//------------------------------------------------------------------------------------------
__global__ void normalizeDensity(int* _activity,
                                 int* _cellParticleType,
                                 real_t* _sortedDensity,
                                 real_t* _sortedNormalizedDensity,
                                 real_t* _sortedPressure,
                                 real4_t* _sortedPosition,
                                 int* _particleUnsortedIndex,
                                 int* _cellStartIndex,
                                 int* _cellEndIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    real4_t pos_i = FETCH(_sortedPosition, index);

    int3 cellPos = calcGridPos(pos_i);


    int cellIndex = calcGridHash(cellPos);

    if(cellIndex < 0)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_cellParticleType[cellIndex] == CELL_SPH && _activity[pIndex] != ACTIVE)
    {
        return;
    }

    real4_t diff_pos;
    real_t dist_sq;

    int neightborCellPos_x, neightborCellPos_y, neightborCellPos_z;
    int neighborCellHash, startIndex, endIndex, j;

    real_t tmp = 0.0;

    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                neightborCellPos_x = (int) cellPos.x + x;
                neightborCellPos_y = (int) cellPos.y + y;
                neightborCellPos_z = (int) cellPos.z + z;

                if (neightborCellPos_x < 0 ||
                        neightborCellPos_y < 0 ||
                        neightborCellPos_z < 0 ||
                        neightborCellPos_x >= simParams.num_cell_x ||
                        neightborCellPos_y >= simParams.num_cell_y ||
                        neightborCellPos_z >= simParams.num_cell_z )
                {
                    continue;
                }

                neighborCellHash = calcGridHash(
                                       make_int3(neightborCellPos_x, neightborCellPos_y,
                                                 neightborCellPos_z));

                startIndex = FETCH(_cellStartIndex, neighborCellHash);

                if (startIndex != 0xffffffff)       // cell is not empty
                {
                    endIndex = FETCH(_cellEndIndex, neighborCellHash);

                    for (j = startIndex; j < endIndex; ++j)
                    {

                        diff_pos = pos_i - FETCH(_sortedPosition, j);
                        dist_sq = diff_pos.x * diff_pos.x
                                  + diff_pos.y * diff_pos.y
                                  + diff_pos.z * diff_pos.z;

                        tmp += (kernel_poly6(dist_sq / simParams.sph_kernel_smooth_length_squared) *
                                simParams.sph_kernel_poly6 * simParams.sph_particle_mass) /
                               _sortedDensity[j];

                    }
                }
            }
        }
    }


    if (tmp > 0.0)
    {
        tmp = _sortedDensity[index] / tmp;
    }



    // calculate pressure
    real_t rho = tmp / simParams.sph_rest_density;
    _sortedPressure[index] = SPH_Vs2 / 7.0 * (powf(rho, 7) - 1);


    _sortedNormalizedDensity[index] = tmp;
}

//------------------------------------------------------------------------------------------
__global__ void normalizeDensityBoundary(int* _activity,
        int* _cellParticleType,
        real_t* _sortedDensity,
        real_t* _sortedNormalizedDensity,
        real_t* _sortedPressure,
        real4_t* _sortedPosition,
        real4_t* _BPPosition,
        int* _particleUnsortedIndex)
{
    int index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_sph_particle)
    {
        return;
    }

    real4_t pos_i = FETCH(_sortedPosition, index);

    int3 cellPos = calcGridPos(pos_i);

    if(!isBoundaryCell(cellPos))
    {
        return;
    }

    int cellIndex = calcGridHash(cellPos);

    if(cellIndex < 0)
    {
        return;
    }

    int pIndex = _particleUnsortedIndex[index];

    if(_cellParticleType[cellIndex] == CELL_SPH && _activity[pIndex] != ACTIVE)
    {
        return;
    }

    real_t tmp = 0;
    real4_t pos_i2;
    int startIndex = 0;
    int endIndex = 0;

    // bottom plane
    if (cellPos.y == 0)
    {
        startIndex = simParams.boundaryPlaneBottomIndex;
        endIndex = startIndex + simParams.boundaryPlaneBottomSize;


        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);

    }


    // top plane
    if (cellPos.y == (simParams.num_cell_y - 1))
    {
        startIndex = simParams.boundaryPlaneTopIndex;
        endIndex = startIndex + simParams.boundaryPlaneTopSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);

    }

// left side plane
    if (cellPos.x == 0)
    {
        startIndex = simParams.boundaryPlaneLeftSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneLeftSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// right side plane
    if (cellPos.x == (simParams.num_cell_x - 1))
    {
        startIndex = simParams.boundaryPlaneRightSideIndex;
        endIndex = startIndex + simParams.boundaryPlaneRightSideSize;

        pos_i2.x = pos_i.x;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z - cellPos.z * simParams.cell_size;

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.z > 0)
        {
            pos_i2.z += simParams.cell_size;
        }

        if (cellPos.z == (simParams.num_cell_z - 1))
        {
            pos_i2.z += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// back plane
    if (cellPos.z == 0)
    {
        startIndex = simParams.boundaryPlaneBackIndex;
        endIndex = startIndex + simParams.boundaryPlaneBackSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }

// front plane
    if (cellPos.z == (simParams.num_cell_z - 1))
    {
        startIndex = simParams.boundaryPlaneFrontIndex;
        endIndex = startIndex + simParams.boundaryPlaneFrontSize;

        pos_i2.x = pos_i.x - cellPos.x * simParams.cell_size;
        pos_i2.y = pos_i.y - cellPos.y * simParams.cell_size;
        pos_i2.z = pos_i.z;

        if (cellPos.x > 0)
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.x == (simParams.num_cell_x - 1))
        {
            pos_i2.x += simParams.cell_size;
        }

        if (cellPos.y > 0)
        {
            pos_i2.y += simParams.cell_size;
        }

        if (cellPos.y == (simParams.num_cell_y - 1))
        {
            pos_i2.y += simParams.cell_size;
        }

        tmp += densityBP(startIndex, endIndex, pos_i2, _BPPosition);
    }



    if (tmp > 0.0)
    {
        tmp *= (simParams.sph_kernel_poly6 * simParams.sph_particle_mass /
                simParams.sph_rest_density);
        tmp = 1.0 / (1.0 / _sortedNormalizedDensity[index] + tmp / _sortedDensity[index]);
        // calculate pressure
        real_t rho = tmp / simParams.sph_rest_density;
        _sortedPressure[index] = SPH_Vs2 / 7.0 * (powf(rho, 7.0) - 1);
//        printf("%d: norm = %f\n", index, _sortedNormalizedDensity[index]);

//        must be = (replace it)
        _sortedNormalizedDensity[index] = tmp;
    }

}


//------------------------------------------------------------------------------------------
__global__ void updatePDStretchNeighborList(real_t* _stretch,
                                            real_t* _newStretch,
                                            struct Clist* _pdNeighborList)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_pd_particle)
    {
        return;
    }

    _stretch[index] = _newStretch[index];
    _pdNeighborList[index].plist_top = -1;

    if(index + simParams.num_pd_particle < simParams.num_clists)
    {
        _pdNeighborList[index + simParams.num_pd_particle].plist_top = -1;
    }

}



#endif // SIMULATOR_CUH

