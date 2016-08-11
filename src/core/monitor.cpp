//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include "utilities.h"
#include "monitor.h"
#include <ctime>

//------------------------------------------------------------------------------------------
Monitor::Monitor()
{
    checkCudaErrors(cudaEventCreate(&eStart));
    checkCudaErrors(cudaEventCreate(&eStop));

    event2NameMaps[COLLECT_PARTICLES] = "Collect particles";
    event2NameMaps[UPDATE_VELOCITY_POSITION] = "Update velocity and position";
    event2NameMaps[CALCULATE_SPH_PARTICLE_TIMESTEP] = "Calculate particle timestep";
    event2NameMaps[CALCULATE_SPH_DENSITY_PRESSURE] = "Calculate SPH density and pressure";
    event2NameMaps[LIMIT_SPH_TIMESTEP] = "Limit SPH timestep at SPH-PD boundary";
    event2NameMaps[CALCULATE_SPH_FORCE] = "Calculate SPH particle force";
    event2NameMaps[CALCULATE_PERIDYNAMICS_FORCE] = "Calculate Peridynamics particle force";
    event2NameMaps[CALCULATE_SPH_PD_FORCE] = "Calculate SPH-PD interaction force";
    event2NameMaps[SOLVE_LINEAR_EQUATION] = "Solve linear equation by preconditioner CG";


    TRUE_OR_DIE(event2NameMaps.size() == NUM_EVENTS,
                "Ohh, you've omitted to initialize some events....");

    setStream(0);
    resetTimer();
}

//------------------------------------------------------------------------------------------
const string Monitor::PADDING = "   ";

//------------------------------------------------------------------------------------------
Monitor::~Monitor()
{
    checkCudaErrors(cudaEventDestroy(eStart));
    checkCudaErrors(cudaEventDestroy(eStop));
}

//------------------------------------------------------------------------------------------
void Monitor::setStream(cudaStream_t _stream)
{
    stream = _stream;
}

//------------------------------------------------------------------------------------------
void Monitor::startTimer()
{
    checkCudaErrors(cudaEventRecord(eStart, stream));
}

//------------------------------------------------------------------------------------------
void Monitor::recordToEvent(Monitor::Events _event)
{
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, eStart, eStop));
    event2TimeMaps[_event] += elapsedTime;
    elapsedTime = 0.0f;
}

//------------------------------------------------------------------------------------------
float Monitor::getElapsedTime()
{
    return elapsedTime;
}

//------------------------------------------------------------------------------------------
void Monitor::resetTimer()
{
    for(std::map<Events, float>::iterator ptr = event2TimeMaps.begin();
        ptr != event2TimeMaps.end(); ++ptr)
    {
        ptr->second = 0.0f;
    }
}

//------------------------------------------------------------------------------------------
void Monitor::printSimulationTime()
{
    float totalTime = 0;

    for(std::map<Events, float>::iterator ptr = event2TimeMaps.begin();
        ptr != event2TimeMaps.end(); ++ptr)
    {
        cout << Monitor::PADDING << event2NameMaps[ptr->first] << ": " << ptr->second << endl;
        totalTime += (float) ptr->second;
    }

    cout << Monitor::PADDING << "Total time: " << totalTime << endl;
}

//------------------------------------------------------------------------------------------
void Monitor::recordEvent(const char* _event)
{
    time_t currentTime = std::time(NULL);
    struct tm* localTime;
    localTime = std::localtime(&currentTime);  // Convert the current time to the local time

    cout << "[" << localTime->tm_hour << ":" << localTime->tm_min << ":" << localTime->tm_sec
         << "] " << _event << endl;
}

//------------------------------------------------------------------------------------------
void Monitor::blankLine()
{
    cout << endl;
}

