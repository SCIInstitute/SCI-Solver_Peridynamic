//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <fstream>
#include <string>
#include <math.h>
#include <sstream>

#include "definitions.h"
#include "utilities.h"
#include "parameters.h"
#include "monitor.h"

using namespace std;

//------------------------------------------------------------------------------------------
SystemParameter::SystemParameter(const char* _paramFile)
{
    initParameters();
    loadParametersFromFile(_paramFile);
}

//------------------------------------------------------------------------------------------
SimulationParameters& SystemParameter::getSimulationParameters()
{
    return simParams;
}

//------------------------------------------------------------------------------------------
RunningParameters& SystemParameter::getRunningParameters()
{
    return runningParams;
}

//------------------------------------------------------------------------------------------
void SystemParameter::initParameters()
{
    simParams.num_total_particle = 0;
    simParams.num_sph_particle = 0;
    simParams.num_pd_particle = 0;
    simParams.num_cell_x = 1;
    simParams.num_cell_y = 1;
    simParams.num_cell_z = 1;
    simParams.num_cells = 1;

    simParams.sph_kernel_coeff = 4.001f;
    simParams.pd_kernel_coeff = 6.001;
    simParams.sph_particle_mass = 1.0f;
    simParams.pd_particle_mass = 1.0f;
    simParams.sph_sph_viscosity = 1e-5;
    simParams.sph_pd_slip = 1.0;
    simParams.sph_boundary_viscosity = 1e-5;

    simParams.boundary_min_x = 0.0f;
    simParams.boundary_min_y = 0.0f;
    simParams.boundary_min_z = 0.0f;
    simParams.boundary_max_x = 1.0f;
    simParams.boundary_max_y = 1.0f;
    simParams.boundary_max_z = 1.0f;

    simParams.boundaryPlaneBottomIndex = 0;
    simParams.boundaryPlaneTopIndex = 0;
    simParams.boundaryPlaneFrontIndex = 0;
    simParams.boundaryPlaneBackIndex = 0;
    simParams.boundaryPlaneLeftSideIndex = 0;
    simParams.boundaryPlaneRightSideIndex = 0;



    // the values below are set up later via object geometry
    simParams.sph_kernel_smooth_length = 0.0f;
    simParams.sph_kernel_smooth_length_squared = 0.0f;
    simParams.sph_particle_radius = 0.0f;
    simParams.sph_kernel_poly6 = 0.0f;
    simParams.sph_kernel_spiky = 0.0f;
    simParams.sph_rest_density = 0.0f;
    simParams.pd_particle_radius = 0.0f;
    simParams.pd_horizon = 0.0f;
    simParams.pd_C_times_V = 0.0f;

    simParams.clockScale = 0.0f;
}

//------------------------------------------------------------------------------------------
void SystemParameter::loadParametersFromFile(const char* _paramFile)
{
    std::ifstream inFile(_paramFile);
    TRUE_OR_DIE(inFile.is_open(), "Could not open parameter file.");

    std::string line;
    std::string paramName, paramValue;

    while (std::getline(inFile, line))
    {
        line.erase(line.find_last_not_of(" \n\r\t") + 1);

        if(line == "")
        {
            continue;
        }

        if(line.find("//") != std::string::npos)
        {
            continue;
        }

        std::istringstream iss(line);
        iss >> paramName >> paramValue;


        /////////////////////////////////////////////////////////////////
        // simulation parameters

        std::cout << Monitor::PADDING << paramName << ": " << paramValue << std::endl;


        if(paramName == "num_sph_particle")
        {
            simParams.num_sph_particle = atoi(paramValue.c_str());
        }

        if(paramName == "num_pd_particle")
        {
            simParams.num_pd_particle = atoi(paramValue.c_str());
        }


        if(paramName == "radius_pd_over_sph")
        {
            simParams.pd_particle_radius = atof(paramValue.c_str());
        }


        if(paramName == "boundary_min_x")
        {
            simParams.boundary_min_x = atof(paramValue.c_str());
        }

        if(paramName == "boundary_min_y")
        {
            simParams.boundary_min_y = atof(paramValue.c_str());
        }


        if(paramName == "boundary_min_z")
        {
            simParams.boundary_min_z = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_x")
        {
            simParams.boundary_max_x = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_y")
        {
            simParams.boundary_max_y = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_z")
        {
            simParams.boundary_max_z = atof(paramValue.c_str());
        }


        if(paramName == "sph_kernel_coeff")
        {
            simParams.sph_kernel_coeff = atof(paramValue.c_str());
        }

        if(paramName == "pd_kernel_coeff")
        {
            simParams.pd_kernel_coeff = atof(paramValue.c_str());
        }


        if(paramName == "sph_particle_mass")
        {
            simParams.sph_particle_mass = atof(paramValue.c_str());
        }

        if(paramName == "pd_particle_mass")
        {
            simParams.pd_particle_mass = atof(paramValue.c_str());
        }

        if(paramName == "sph_sph_viscosity")
        {
            simParams.sph_sph_viscosity = atof(paramValue.c_str());
        }


        if(paramName == "sph_boundary_viscosity")
        {
            simParams.sph_boundary_viscosity = atof(paramValue.c_str());
        }

        if(paramName == "sph_pd_slip")
        {
            simParams.sph_pd_slip = atof(paramValue.c_str());
        }




        /////////////////////////////////////////////////////////////////
        // cpu parameter only
        if(paramName == "adaptive_integration")
        {
            runningParams.adaptive_integration = atoi(paramValue.c_str());
        }


        if(paramName == "gpu_device")
        {
            runningParams.gpu_device = atoi(paramValue.c_str());
        }


        if(paramName == "steps_per_frame")
        {
            runningParams.step_per_frame =  atoi(paramValue.c_str());
        }


        if(paramName == "tube_radius")
        {
            runningParams.tube_radius =  atoi(paramValue.c_str());
        }

        if(paramName == "wall_thickness")
        {
            runningParams.wall_thickness =  atoi(paramValue.c_str());
        }

        if(paramName == "mesh_translation_x")
        {
            runningParams.mesh_translation_x = atof(paramValue.c_str());
        }

        if(paramName == "mesh_translation_y")
        {
            runningParams.mesh_translation_y = atof(paramValue.c_str());
        }

        if(paramName == "sph_initial_velocity")
        {
            runningParams.sph_initial_velocity = atof(paramValue.c_str());
        }

        if(paramName == "mesh_translation_z")
        {
            runningParams.mesh_translation_z = atof(paramValue.c_str());
        }

        if(paramName == "pd_stretch_limit_s0")
        {
            runningParams.pd_stretch_limit_s0 = atof(paramValue.c_str());
        }

        if(paramName == "pd_initial_velocity")
        {
            runningParams.pd_initial_velocity = atof(paramValue.c_str());
        }




        if(paramName == "steps_per_state")
        {
            runningParams.step_per_state =  atoi(paramValue.c_str());
        }

        if(paramName == "start_step")
        {
            runningParams.start_step = atoi(paramValue.c_str());
        }

        if(paramName == "final_step")
        {
            runningParams.final_step = atoi(paramValue.c_str());
        }




        if(paramName == "dump_density")
        {
            runningParams.dump_density = atoi(paramValue.c_str());
        }

        if(paramName == "dump_velocity")
        {
            runningParams.dump_velocity = atoi(paramValue.c_str());
        }

        if(paramName == "dump_activity")
        {
            runningParams.dump_activity = atoi(paramValue.c_str());
        }


        if(paramName == "dump_pd_bond_list_top")
        {
            runningParams.dump_pd_bond_list_top = atoi(paramValue.c_str());
        }





        if(paramName == "obj_file")
        {
            strcpy(runningParams.obj_file, paramValue.c_str());
            std::cout << "obj_file:  " << runningParams.obj_file << std::endl;
        }

        if(paramName == "saving_path")
        {
            strcpy(runningParams.saving_path, paramValue.c_str());
            std::cout << "save:  " << runningParams.saving_path << std::endl;
        }

        if(paramName == "integrator")
        {
            if(paramValue == "implicit_euler")
            {
                runningParams.integrator = IMPLICIT_EULER;
                std::cout << "Integrator: Implicit Euler" << std::endl;
            }
            else
            {
                runningParams.integrator = NEWMARK_BETA;
                std::cout << "Integrator: Newmark Beta" << std::endl;
            }
        }

    }

    inFile.close();
    calculateDependentParameters();

}


//------------------------------------------------------------------------------------------
void SystemParameter::calculateDependentParameters()
{
    simParams.num_total_particle = simParams.num_sph_particle + simParams.num_pd_particle;

    simParams.num_cell_x = NUM_CELL_X;
    simParams.num_cell_y = NUM_CELL_Y;
    simParams.num_cell_z = NUM_CELL_Z;
    simParams.num_cells = NUM_CELLS;

    simParams.cell_size = fmaxf(
                              fmaxf((simParams.boundary_max_x - simParams.boundary_min_x) / NUM_CELL_X,
                                    (simParams.boundary_max_y - simParams.boundary_min_y) / NUM_CELL_Y),
                              (simParams.boundary_max_z - simParams.boundary_min_z) / NUM_CELL_Z);
    simParams.sph_kernel_smooth_length = simParams.cell_size;
    simParams.sph_particle_radius = simParams.cell_size / simParams.sph_kernel_coeff;

    simParams.pd_time_step = TIMESTEP_PD;
    simParams.pd_particle_radius *= simParams.sph_particle_radius;
    simParams.pd_horizon = simParams.pd_particle_radius * simParams.pd_kernel_coeff;
    simParams.pd_C_times_V = (18.0 / PI / 2.0 / 1296.0 / simParams.pd_particle_radius);
//                                 simParams.pd_particle_mass;


//    double d = simParams.sph_kernel_smooth_length;
//    double d2 = d * d;
//    double d6 = d2 * d2 * d2;
//    double d9 = d6 * d2 * d;
//    double k_poly6 = 315.0 / 64.0 / 3.14159265358979323846264 / d9;
//    double k_spiky = 45.0 / 3.14159265358979323846264 / d6;
    double k_poly6 = 315.0 / 64.0 / PI;
    double k_spiky = 45.0 / PI;
    simParams.sph_kernel_poly6 = (real_t)k_poly6;
    simParams.sph_kernel_spiky = (real_t)k_spiky;


    ////////////////////////////////////////////////////////////////////////////////
    // print out
    std::cout << Monitor::PADDING << "Num. cells: " << simParams.num_cells <<
              "(" << simParams.num_cell_x << "x" << simParams.num_cell_y << "x" << simParams.num_cell_z
              << ")" << std::endl;
    std::cout << Monitor::PADDING << "Radius of SPH particle: " <<
              simParams.sph_particle_radius << std::endl;
    std::cout << Monitor::PADDING << "Radius of Peridynamics particle: " <<
              simParams.pd_particle_radius << std::endl;
    std::cout << Monitor::PADDING << "Kernel smooth length: " <<
              simParams.sph_kernel_smooth_length << std::endl;
    std::cout << Monitor::PADDING << "Kernel Poly6  coefficient: " <<
              simParams.sph_kernel_poly6 << std::endl;
    std::cout << Monitor::PADDING << "Kernel Spiky coefficient: " <<
              simParams.sph_kernel_spiky << std::endl;
    std::cout << Monitor::PADDING << "Peridynamics C*V coefficient: " <<
              simParams.pd_C_times_V << std::endl;
    std::cout << Monitor::PADDING << "Peridynamics wall thickness: " <<
              runningParams.wall_thickness << " particles" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////
    // now, scale everything to peridynamics framework:
    // boundary, particle radius, cell_size, kernel smooth length
    simParams.scaleFactor = 2 * simParams.pd_particle_radius / PERIDYNAMICS_MIN_DISTANCE;

    simParams.boundary_min_x /= simParams.scaleFactor;
    simParams.boundary_max_x /= simParams.scaleFactor;
    simParams.boundary_min_y /= simParams.scaleFactor;
    simParams.boundary_max_y /= simParams.scaleFactor;
    simParams.boundary_min_z /= simParams.scaleFactor;
    simParams.boundary_max_z /= simParams.scaleFactor;

    simParams.sph_particle_radius /= simParams.scaleFactor;
    simParams.pd_particle_radius /= simParams.scaleFactor;
    simParams.sph_pd_min_distance = simParams.sph_particle_radius +
                                    simParams.pd_particle_radius;


    simParams.cell_size /= simParams.scaleFactor;

    simParams.pd_horizon /= simParams.scaleFactor;
    simParams.sph_kernel_smooth_length /= simParams.scaleFactor;
    simParams.sph_kernel_smooth_length_squared = simParams.sph_kernel_smooth_length *
                                                 simParams.sph_kernel_smooth_length;

    simParams.boundaryPlaneSize = (int) ceil(3 * simParams.cell_size /
                                             (simParams.sph_particle_radius * 2));

    ////////////////////////////////////////////////////////////////////////////////
    // print out
    std::cout << std::endl;
    std::cout << Monitor::PADDING << "Scale factor: " << simParams.scaleFactor <<
              ". AFTER SCALING:"
              << std::endl;

    std::cout << Monitor::PADDING << "Radius of SPH particle: " <<
              simParams.sph_particle_radius << std::endl;
    std::cout << Monitor::PADDING << "Radius of Peridynamics particle: " <<
              simParams.pd_particle_radius << std::endl;
    std::cout << Monitor::PADDING << "Kernel smooth length: " <<
              simParams.sph_kernel_smooth_length << std::endl;
    std::cout << Monitor::PADDING << "Boundary min/max: [" <<
              simParams.boundary_min_x << "," << simParams.boundary_min_y << "," <<
              simParams.boundary_min_z << "] -> " <<
              simParams.boundary_max_x << "," << simParams.boundary_max_y << "," <<
              simParams.boundary_max_z << "]" << std::endl;

}


//------------------------------------------------------------------------------------------
void SystemParameter::printSystemParameters()
{

}

