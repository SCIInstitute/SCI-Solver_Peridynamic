//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <stdint.h>

#include "definitions.h"

#ifndef PARAMETERS_H
#define PARAMETERS_H


struct SimulationParameters
{
    // read from file
    int num_total_particle;
    int num_sph_particle;
    int num_pd_particle;

    int num_cell_x;
    int num_cell_y;
    int num_cell_z;
    int num_cells;
    int num_clists;

    real_t cell_size;

    real_t sph_kernel_coeff;
    real_t pd_kernel_coeff;
    real_t sph_particle_mass;
    real_t pd_particle_mass;
    real_t sph_sph_viscosity;
    real_t sph_boundary_viscosity;
    real_t sph_pd_slip;

    real_t boundary_min_x;
    real_t boundary_min_y;
    real_t boundary_min_z;
    real_t boundary_max_x;
    real_t boundary_max_y;
    real_t boundary_max_z;

    real_t pd_min_x;
    real_t pd_min_y;
    real_t pd_min_z;
    real_t pd_max_x;
    real_t pd_max_y;
    real_t pd_max_z;

    int boundaryPlaneSize;
    int boundaryPlaneBottomIndex;
    int boundaryPlaneBottomSize;
    int boundaryPlaneTopIndex;
    int boundaryPlaneTopSize;
    int boundaryPlaneFrontIndex;
    int boundaryPlaneFrontSize;
    int boundaryPlaneBackIndex;
    int boundaryPlaneBackSize;
    int boundaryPlaneLeftSideIndex;
    int boundaryPlaneLeftSideSize;
    int boundaryPlaneRightSideIndex;
    int boundaryPlaneRightSideSize;

    int pd_max_num_bonds;
    int pd_time_step;

    // the values below are set up later via object geometry
    real_t sph_kernel_smooth_length;
    real_t sph_kernel_smooth_length_squared;
    real_t sph_particle_radius;
    real_t sph_kernel_poly6;
    real_t sph_kernel_spiky;
    real_t sph_rest_density;

    real_t pd_particle_radius;
    real_t pd_horizon;
    real_t pd_C_times_V;
    real_t sph_pd_min_distance;

    real_t clockScale;
    real_t scaleFactor;


};


struct RunningParameters
{
    RunningParameters():
        gpu_device(0),
        step_per_frame(1),
        final_step(1),
        dump_density(0),
        dump_velocity(0),
        wall_thickness(20),
        mesh_translation_x(0.0),
        mesh_translation_y(0.0),
        mesh_translation_z(0.0),
        sph_initial_velocity(1.0),
        integrator(IMPLICIT_EULER)
    {}

    int adaptive_integration;
    int gpu_device;
    int step_per_frame;
    int step_per_state;
    int start_step;
    int final_step;

    int wall_thickness;
    int tube_radius;
    float emiter_x;
    float emiter_y;
    float sph_initial_velocity;
    float mesh_translation_x;
    float mesh_translation_y;
    float mesh_translation_z;
    float wall_translation_z;

    float pd_stretch_limit_s0;
    float pd_initial_velocity;

    int dump_activity;
    int dump_density;
    int dump_velocity;
    int dump_pd_bond_list_top;

    char obj_file[512];
    char saving_path[512];

    Integrator integrator;
};

enum Activity
{
    ACTIVE = 0,
    SEMI_ACTIVE,
    INACTIVE,
    INVISIBLE,
    NUM_ACTIVITY_MODE
};

class SystemParameter
{
public:
    SystemParameter(const char* _paramFile);

    SimulationParameters& getSimulationParameters();
    RunningParameters& getRunningParameters();

private:
    void initParameters();
    void loadParametersFromFile(const char* _paramFile);
    void calculateDependentParameters();
    void printSystemParameters();

    SimulationParameters simParams;
    RunningParameters runningParams;
};

#endif // PARAMETERS_H
