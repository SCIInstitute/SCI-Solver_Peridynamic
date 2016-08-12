//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef SIMULATIONMEMORY_H
#define SIMULATIONMEMORY_H

#include <stdint.h>
#include "helper_cuda.h"
#include <map>

#include "definitions.h"
#include "utilities.h"
#include "parameters.h"

class MemoryManager
{
public:
    enum MemoryLocation
    {
        HOST_LOCATION,
        DEVICE_LOCATION
    };

    enum Variables
    {
        SPH_TIMESTEP = 0,
        SPH_ACTIVITY,
        SPH_VALIDITY,

        SPH_POSITION,
        SPH_VELOCITY,
        SPH_FORCE,
        SPH_SORTED_POSITION,
        SPH_SORTED_VELOCITY,
        SPH_SORTED_DENSITY,
        SPH_SORTED_DENSITY_NORMALIZED,
        SPH_SORTED_PRESSURE,

        // boundary particles
        SPH_BOUNDARY_POSITION,

        // peridynamics particles
        PD_ACTIVITY,
        PD_POSITION,
        PD_POSITION_BACKUP,
        PD_VELOCITY,
        PD_VELOCITY_BACKUP,
        PD_FORCE,
        PD_SORTED_POSITION,
        PD_SORTED_VELOCITY,
        PD_ORIGINAL_POSITION,
        PD_ORIGINAL_STRETCH,
        PD_ORIGINAL_BOND_LIST_TOP,
        PD_STRETCH,
        PD_NEW_STRETCH,
        PD_BOND_LIST,
        PD_BOND_LIST_TOP,
        PD_BOND_LIST_TOP_BACKUP,
        PD_CLIST,

        PD_SYSTEM_MATRIX,
        PD_SYSTEM_VECTOR,
        PD_SYSTEM_SOLUTION,
        PD_HAS_BROKEN_BOND,


        // cells
        CELL_PARTICLE_TYPE,


        SPH_PARTICLE_TO_CELL_HASH,
        SPH_PARTICLE_UNSORTED_INDEX,
        SPH_CELL_START_INDEX,
        SPH_CELL_END_INDEX,

        PD_PARTICLE_TO_CELL_HASH,
        PD_PARTICLE_UNSORTED_INDEX,
        PD_CELL_START_INDEX,
        PD_CELL_END_INDEX,




        NUM_VARIABLES
    };

    enum VariableTypes
    {
        REAL_TYPE,
        REAL4_TYPE,
        INT32_TYPE,
        STRUCT,
        NUM_TYPES
    };

    MemoryManager(SimulationParameters &simParams_);
    ~MemoryManager();

    void uploadToDevice(Variables _variable, bool _scale);
    void uploadAllArrayToDevice(bool _scale);
    void downloadAllArrayFromDevice();
    void downloadFromDevice(Variables _variable);
    void printArray(Variables _variable, int _size = 0);
    void printHostArray(Variables _variable, int _size = 0);
    void printPositiveIntegerArray(Variables _variable, int _size = 0);
    void* getHostPointer(Variables _variable);
    void* getDevicePointer(Variables _variable);
    size_t getArraySize(Variables _variable);
    char* getVariableName(Variables _variable);
    void countMemory();
    void backupBondListTopIndex();
    void restoreBondListTopIndex();

    void backupPDPosition();
    void backupPDVelocity();
    void restorePDPosition();
    void restorePDVelocity();
    void transferData(Variables dest_var, Variables source_var);

private:
    void allocateHostMemory();
    void allocateDeviceMemory();
    void scaleParticle(real4_t* _parPos, int _numParticles, real_t _scaleFactor);

    void allocateDeviceArray(void** devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeDeviceArray(void* devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    SimulationParameters &simParams_;
    std::map<Variables, void*> hostPointerMap;
    std::map<Variables, void*> devicePointerMap;
    std::map<Variables, size_t> sizeMap;
    std::map<Variables, VariableTypes> typeMap;
    std::map<Variables, char*> variable2NameMap;
};

#endif // SIMULATIONMEMORY_H
