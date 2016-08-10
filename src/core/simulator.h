//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <helper_math.h>
#include <math_constants.h>

//#include "cg_solver.cuh"
#include "parameters.h"
#include "memory_manager.h"
#include "monitor.h"
#include "dataIO.h"

//------------------------------------------------------------------------------------------
#define DEFAULT_BLOCK_SIZE 256

struct KernelSize
{
    int nthreads;
    int nblocks;
};


//------------------------------------------------------------------------------------------
class Simulator
{
public:
    Simulator(int deviceID, SimulationParameters& simParams,
              RunningParameters& runningParams);
    ~Simulator();

    void makeReady();
    void advance();
    void printSimulationTime();


    bool finished();
    bool CGConvergence();

    int get_current_step() const;
    int* get_sph_activity();
    real4_t* get_sph_position();
    real4_t* get_sph_velocity();
    real4_t* get_sph_boundary_pos();

    int* get_pd_activity();
    real4_t* get_pd_position();
    real4_t* get_pd_velocity();

    DataIO& dataIO();

protected:
    void integrateSPH();
    bool integratePeridynamicsImplicitEuler();
    bool integratePeridynamicsNewmarkBeta();

    void printDebugInfoLinearSystem();
    void solveLinearSystem();


    void initPeridynamicsBonds();
    void calculateKernelOccupancy();
    void calculateSPHActivity();
    void collectParticles();
    void limitSPHTimestep();
    void calculateSPHDensityPressure();
    void updateSimParamsToDevice();

    ////////////////////////////////////////////////////////////////////////////////
    void mapDeviceMemory();
    int* sph_timestep;
    int* sph_validity;
    int* sph_activity;

    real4_t* sph_position;
    real4_t* sph_velocity;
    real4_t* sph_force;
    real4_t* sph_sorted_pos;
    real4_t* sph_sorted_vel;
    real_t* sph_sorted_density;
    real_t* sph_sorted_density_normalized;
    real_t* sph_sorted_pressure;
    real4_t* sph_boundary_pos;

    int* pd_activity;
    real4_t* pd_position;
    real4_t* pd_velocity;
    real4_t* pd_force;
    real4_t* pd_sorted_pos;
    real4_t* pd_sorted_vel;
    real4_t* pd_original_pos;
    real_t* pd_original_stretch;
    int* pd_original_bond_list_top;
    real_t* pd_stretch;
    real_t* pd_new_stretch;
    int* pd_bond_list_top;
    int* pd_bond_list;
    Clist* pd_clist;

    Mat3x3* pd_system_matrix;
    real4_t* pd_system_vector;
    real4_t* pd_system_solution;
    int* pd_has_broken_bond;


    int* cell_type;


    int* sph_cell_hash;
    int* sph_unsorted_index;
    int* sph_cell_start_index;
    int* sph_cell_end_index;

    int* pd_cell_hash;
    int* pd_unsorted_index;
    int* pd_cell_start_index;
    int* pd_cell_end_index;

    ////////////////////////////////////////////////////////////////////////////////
    int current_step_;
    int current_substep_;
    bool CG_convergent_;

    KernelSize kernelSPH_;
    KernelSize kernelPD_;
    KernelSize kernelGrid_;

    SimulationParameters& hostSimParams_;
    RunningParameters& runningParams_;
    Monitor monitor_;
    MemoryManager simMemory_;
    DataIO dataIO_; // dataIO must be declared after memory object

};


#endif // SIMULATOR_H
