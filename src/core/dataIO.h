//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef DATAWRITER_H
#define DATAWRITER_H

#include <map>

#include "memory_manager.h"
#include "parameters.h"

using namespace std;

class DataIO
{
public:
    DataIO(SimulationParameters &simParams, RunningParameters &runningParams, MemoryManager& simMemory);
    ~DataIO();

    void readSimulationParameter(SimulationParameters& _simParams);
    void writeSimulationParameter();
    void getDataAndWrite();
    void getDataAndWrite(MemoryManager::Variables _variable);
    void saveState(size_t _timestep);
    void loadState(size_t _timestep);

    void newFrame();

    size_t savedFrame;

private:
    void createOutputFolders();

    SimulationParameters& simParams_;
    RunningParameters& runningParams_;
    MemoryManager& simMemory_;
    const char* savingPath;
    std::map<MemoryManager::Variables, int> savingMap;
    bool writtenSimulationParameter;
};

#endif // DATAWRITER_H
