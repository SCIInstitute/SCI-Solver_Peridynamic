//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef MONITOR_H
#define MONITOR_H

#include <vector_types.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <map>
#include <string>
#include <iostream>

using namespace std;

class Monitor
{
public:
    enum Events
    {
        COLLECT_PARTICLES = 0,
        UPDATE_VELOCITY_POSITION,
        CALCULATE_SPH_PARTICLE_TIMESTEP,
        CALCULATE_SPH_DENSITY_PRESSURE,
        LIMIT_SPH_TIMESTEP,
        CALCULATE_SPH_FORCE,
        CALCULATE_PERIDYNAMICS_FORCE,
        CALCULATE_SPH_PD_FORCE,
        SOLVE_LINEAR_EQUATION,

        NUM_EVENTS
    };

    static const string PADDING;
    Monitor();
    ~Monitor();

    void setStream(cudaStream_t _stream);
    void startTimer();
    void recordToEvent(Events _event);

    float getElapsedTime();
    void resetTimer();
    void printSimulationTime();
    static void recordEvent(const char* _event);
    static void blankLine();

    cudaEvent_t eStart, eStop;
    cudaStream_t stream;
private:
    map<Events, float> event2TimeMaps;
    map<Events, string> event2NameMaps;
    float elapsedTime;
};

#endif // MONITOR_H
