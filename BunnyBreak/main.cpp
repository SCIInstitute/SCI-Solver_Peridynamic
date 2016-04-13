//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#include <iostream>

#include <PeriCUDAEngine/utilities.h>
#include <PeriCUDAEngine/parameters.h>
#include <PeriCUDAEngine/memory_manager.h>
#include <PeriCUDAEngine/simulator.h>
#include "scene.h"

using namespace std;
//------------------------------------------------------------------------------------------
SystemParameter* sysParams = NULL;
Scene* scene = NULL;
Simulator* simulator = NULL;

//------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        cout << "Usage: ./BunnyBreak <parameter_file>" << endl;
        exit(EXIT_FAILURE);
    }

    /////////////////////////////////////////////////////////////////
    // load parameter from disk
    Monitor::recordEvent("Load system parameters");
    sysParams = new SystemParameter(argv[1]);
    RunningParameters& runningParams = sysParams->getRunningParameters();
    SimulationParameters& simParams = sysParams->getSimulationParameters();

    // modify this line to change to another scene
    scene = new Scene(simParams, runningParams, Scene::REGULAR_GRID);

    if(runningParams.integrator == IMPLICIT_EULER)
    {
        Monitor::recordEvent("Create simulator(Implicite Euler for PD)");
    }
    else
    {
        Monitor::recordEvent("Create simulator(Newmark Beta for PD)");
    }

    checkCudaErrors(cudaSetDevice(runningParams.gpu_device));
    simulator = new Simulator(runningParams.gpu_device, simParams, runningParams);

    /////////////////////////////////////////////////////////////////
    // init data from the beginning or load from disk
    if(runningParams.start_step == 0)
    {
        // find the boundary particle first, to have rest density calculated
        scene->initSPHBoundaryParticles(simulator->get_sph_boundary_pos());
        scene->initSPHParticles(simulator->get_sph_activity(),
                                simulator->get_sph_position(),
                                simulator->get_sph_velocity());
        scene->initPeridynamicsParticles(simulator->get_pd_activity(),
                                         simulator->get_pd_position(),
                                         simulator->get_pd_velocity());
    }
    else
    {
        simulator->dataIO().loadState(runningParams.start_step);
    }

    simulator->makeReady();


    /////////////////////////////////////////////////////////////////
    // simulation
    size_t frameStep = 0;
    size_t stateStep = 0;
    char buff[512];

    Monitor::recordEvent("=========================Start simulation=========================");


    // run simulation
    while(!simulator->finished())
    {
        if(frameStep == 0)
        {
            simulator->dataIO().newFrame();
            simulator->dataIO().getDataAndWrite();

            sprintf(buff, "Save frame %d(step %d)", simulator->dataIO().savedFrame,
                    simulator->get_current_step());
            Monitor::recordEvent(buff);
            simulator->printSimulationTime();
        }

        if(stateStep == runningParams.step_per_state)
        {
            simulator->dataIO().saveState(simulator->get_current_step());
            stateStep = 0;
        }

        simulator->advance();
        {
            ++frameStep;
            ++stateStep;
        }

        if(frameStep == runningParams.step_per_frame)
        {
            frameStep = 0;
        }

    }

    cout << "=========================================================================================="
         << endl;
    cout << "Total simulation steps: " << runningParams.final_step << endl;
    cout << "Total saved frames: " << simulator->dataIO().savedFrame << endl;
    cout << "Data saved to: " << runningParams.saving_path << endl;

    if(simulator->CGConvergence())
    {
        cout << "Simulation finshed successfully." << endl;
    }
    else
    {
        cout << "Simulation failed." << endl;
    }

    cout << "=========================================================================================="
         << endl;

    cudaDeviceReset();

    return EXIT_SUCCESS;
}


