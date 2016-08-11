//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef CG_SOLVER_CUH
#define CG_SOLVER_CUH

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <math_constants.h>
#include <thrust/functional.h>

#include <cusp/linear_operator.h>

#include "definitions.h"
#include "parameters.h"
#include "cutil_math_ext.h"

//------------------------------------------------------------------------------------------
// simulation parameters in constant memory:
extern __constant__ SimulationParameters simParams;

//------------------------------------------------------------------------------------------
__device__ __forceinline__ real_t dot(real3_t a, real4_t b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ real_t dot(real4_t a, real3_t b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
//------------------------------------------------------------------------------------------
__device__ __forceinline__ void matrixMulVector(const Mat3x3& A, const real4_t& x,
                                                real4_t& result)
{
    result.x = dot(A.row[0], x);
    result.y = dot(A.row[1], x);
    result.z = dot(A.row[2], x);
    result.w = 0.0;
}

__global__ void multiply(Mat3x3* matrix,
                         real4_t* x,
                         real4_t* result,
                         int* pd_bond_list,
                         int* pd_bond_list_top)
{
    int p = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    int bond_top_index = pd_bond_list_top[p];

    Mat3x3 element = matrix[p];
    real4_t sum = MAKE_REAL4_FROM_REAL(0.0);
    matrixMulVector(element, x[p], sum);

    real4_t xq;

    for (int bond = 0; bond <= bond_top_index; ++bond)
    {
        int bond_index = bond * simParams.num_pd_particle + p;
        element = matrix[bond_index + simParams.num_pd_particle];

        int q = pd_bond_list[bond_index];
        xq = x[q];

        sum.x += dot(element.row[0], xq);
        sum.y += dot(element.row[1], xq);
        sum.z += dot(element.row[2], xq);

    }

    result[p] = sum;
}

//------------------------------------------------------------------------------------------
// r = x - y
__global__ void minusVector(real4_t* x,
                            real4_t* y,
                            real4_t* r)
{
    int p = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    r[p] = x[p] - y[p];
}

//------------------------------------------------------------------------------------------
// result = x + alpha*y
__global__ void plusVector(real4_t* x,
                           real4_t* y,
                           real4_t* result,
                           real_t alpha)
{
    int p = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    result[p] = x[p] + alpha * y[p];
}


//------------------------------------------------------------------------------------------
__device__ __forceinline__ Mat3x3 setZeroMatrix(Mat3x3& matrix)
{
    for(int i = 0; i < 3; ++i)
    {
        matrix.row[i] = MAKE_REAL3_FROM_REAL(0.0);
    }
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void outerProduct(const real4_t& x, const real4_t& y,
                                             Mat3x3& A)
{
    A.row[0] = x.x * MAKE_REAL3_FROM_REAL4(y);
    A.row[1] = x.y * MAKE_REAL3_FROM_REAL4(y);
    A.row[2] = x.z * MAKE_REAL3_FROM_REAL4(y);
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void setIdentityMatrix(Mat3x3& A)
{
    A.row[0] = MAKE_REAL3(1.0, 0.0, 0.0);
    A.row[1] = MAKE_REAL3(0.0, 1.0, 0.0);
    A.row[2] = MAKE_REAL3(0.0, 0.0, 1.0);
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void setDiagonalMatrix(Mat3x3& A, const real_t x)
{
    A.row[0] = MAKE_REAL3(x, 0.0, 0.0);
    A.row[1] = MAKE_REAL3(0.0, x, 0.0);
    A.row[2] = MAKE_REAL3(0.0, 0.0, x);
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void matrixMinus(Mat3x3& A, const Mat3x3& B)
{
    for(int i = 0; i < 3; ++i)
    {
        A.row[i] -= B.row[i];
    }
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void matrixPlus(Mat3x3& A, const Mat3x3& B)
{
    for(int i = 0; i < 3; ++i)
    {
        A.row[i] += B.row[i];
    }
}

//------------------------------------------------------------------------------------------
__device__ __forceinline__ void matrixMultiplyNumber(Mat3x3& A, const real_t x)
{
    for(int i = 0; i < 3; ++i)
    {
        A.row[i] *= x;
    }
}


//------------------------------------------------------------------------------------------
__global__ void initPDSolutions(real4_t* solution)
{
    int index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= simParams.num_pd_particle)
    {
        return;
    }

    solution[index] = MAKE_REAL4_FROM_REAL(0.0);
}




//------------------------------------------------------------------------------------------
__global__ void calculatePDForceDerivative(int* pd_activity,
                                           real4_t* pd_position,
                                           real4_t* pd_force,
                                           real4_t* pd_original_pos,
                                           real_t* pd_original_stretch,
                                           real_t* pd_stretch,
                                           real_t* pd_new_stretch,
                                           int* pd_bond_list,
                                           int* pd_bond_list_top,
                                           Mat3x3* pd_system_matrix,
                                           int* has_broken_bond)
{
    int p = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    if(pd_activity[p] == INVISIBLE)
    {
        return;
    }

    Mat3x3 forceDx;
    Mat3x3 sumForceDx;
    Mat3x3 xijxijt;

    real4_t pos_p = FETCH(pd_position, p);
    real4_t opos_p = pd_original_pos[p];
    real4_t force = MAKE_REAL4_FROM_REAL(0.0);

    int bond_top_index = pd_bond_list_top[p];
    int obond_top_index = bond_top_index;
    real_t stretch = pd_stretch[p];
    real_t smin = 0.0;
    real_t drdivd, ss0, kk, vscale;


    setZeroMatrix(sumForceDx);

    int bond = 0;

    while(bond <= bond_top_index)
    {
        int bond_index = bond * simParams.num_pd_particle + p;
        int q = pd_bond_list[bond_index];

        real4_t eij = FETCH(pd_position, q) - pos_p;
        real_t dij = sqrt(eij.x * eij.x + eij.y * eij.y + eij.z * eij.z);
        eij /= fmaxf(dij, (real_t) SMALL);

        real_t d0 = length(opos_p - pd_original_pos[q]);

        drdivd = dij / d0 - 1.0;



        if (drdivd > fmaxf(stretch, pd_stretch[q]))
        {
            int last_bond_particle = pd_bond_list[bond_top_index * simParams.num_pd_particle + p];
            pd_bond_list[bond_index] = last_bond_particle;
            pd_bond_list[bond_top_index * simParams.num_pd_particle + p] = q;


//            printf("removed bond %d-%d\n", index, j_index);

            --bond_top_index;
            continue; // continue to the next loop without change the walker
        }

        vscale = 1.0;

        if (d0 > (simParams.pd_horizon - simParams.pd_particle_radius))
        {
            vscale = 0.5 + (simParams.pd_horizon - d0) / 2.0 / simParams.pd_particle_radius;
        }

        if (drdivd < smin)
        {
            smin = drdivd;
        }

        kk = drdivd * vscale;
        force += kk * eij;

        outerProduct(eij, eij, xijxijt);
        matrixMultiplyNumber(xijxijt, d0 / dij);

        setDiagonalMatrix(forceDx, 1.0 - d0 / dij);

        matrixPlus(forceDx, xijxijt);

        matrixMultiplyNumber(forceDx, simParams.pd_C_times_V * PD_K / d0);

        matrixPlus(sumForceDx, forceDx);

        matrixMultiplyNumber(forceDx, -1.0);


        // write the force derivative. the first element should be element A_{ii}
        pd_system_matrix[bond_index + simParams.num_pd_particle] = forceDx;

        ++bond;


//        if(p==0 && q==1)
//        {
//            printf("p 0-1: %f, %f, %f,     xij=%f, %f, %f,    %f, %f, %f,    %f, %f, %f\n",
//         eij.x, eij.y, eij.z, forceDx.row[0].x, forceDx.row[0].y, forceDx.row[0].z,
//                    forceDx.row[1].x, forceDx.row[1].y, forceDx.row[1].z,
//                    forceDx.row[2].x, forceDx.row[2].y, forceDx.row[2].z);
//        }
    }


    pd_system_matrix[p] = sumForceDx;


    force *= simParams.pd_C_times_V * PD_K;// _PDStiffness[index];
    ss0 = (10.0 - 9.0 * simParams.clockScale) * pd_original_stretch[p];
    pd_new_stretch[p] = ss0 - alpha_pr * smin;

//    force.y -= (9.8 * simParams.pd_particle_mass);
    force.y -= (simParams.clockScale * 9.8 * simParams.pd_particle_mass);
    force.w = 0;
    pd_force[p] = force;

    if(obond_top_index != bond_top_index)
    {
        pd_bond_list_top[p] = bond_top_index;
        has_broken_bond[0] = 1;
    }
}


//------------------------------------------------------------------------------------------
__global__ void computeAx(int* pd_bond_list,
                          int* pd_bond_list_top,
                          Mat3x3* pd_system_matrix,
                          const real4_t* x, real4_t* y)
{
    int p = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (p >= simParams.num_pd_particle)
    {
        return;
    }

    real4_t xq = x[p];
    Mat3x3 Apq = pd_system_matrix[p];

    real4_t Apx;
    Apx.x = dot(Apq.row[0], xq);
    Apx.y = dot(Apq.row[1], xq);
    Apx.z = dot(Apq.row[2], xq);
    Apx.w = 0.0;

    int bond_list_top = pd_bond_list_top[p];

    __syncthreads();
    for (int bond = 0; bond <= bond_list_top; ++bond)
    {
        int q = pd_bond_list[bond * simParams.num_pd_particle + p];

        xq = x[q];
        Apq = pd_system_matrix[(bond + 1) * simParams.num_pd_particle + p];

        Apx.x += dot(Apq.row[0], xq);
        Apx.y += dot(Apq.row[1], xq);
        Apx.z += dot(Apq.row[2], xq);

    }

    y[p] = Apx;

}

//------------------------------------------------------------------------------------------
class PeridynamicsMatrix: public cusp::linear_operator<real_t, cusp::device_memory>
{
public:
    typedef cusp::linear_operator<real_t, cusp::device_memory> super;

    int* pd_bond_list_;
    int* pd_bond_list_top_;
    Mat3x3* pd_system_matrix_;
    int num_blocks_, num_threads_;

    // constructor
    PeridynamicsMatrix(int* pd_bond_list,
                       int* pd_bond_list_top,
                       Mat3x3* pd_system_matrix,
                       int matrix_size,
                       int num_blocks, int num_threads)
        : super(matrix_size, matrix_size),
          pd_bond_list_(pd_bond_list),
          pd_bond_list_top_(pd_bond_list_top),
          pd_system_matrix_(pd_system_matrix),
          num_blocks_(num_blocks),
          num_threads_(num_threads)
    {}

    // linear operator y = A*x
    template <typename VectorType1,
              typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        // obtain a raw pointer to device memory
        const real_t* x_ptr = thrust::raw_pointer_cast(&x[0]);
        real_t* y_ptr = thrust::raw_pointer_cast(&y[0]);

        computeAx <<< num_blocks_, num_threads_>>> (pd_bond_list_,
                                                    pd_bond_list_top_,
                                                    pd_system_matrix_,
                                                    (const real4_t*)x_ptr,
                                                    (real4_t*)y_ptr);
        getLastCudaError("Kernel execution failed: computeAx");
    }
};




#endif // CG_SOLVER_CUH

