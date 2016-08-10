//------------------------------------------------------------------------------------------
//
//
// Created on: 1/31/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <helper_cuda.h>
#include <math.h>
#include "cutil_math_ext.h"

#include "memory_manager.h"
#include "definitions.h"
#include "monitor.h"

MemoryManager::MemoryManager(SimulationParameters& simParams):
    simParams_(simParams)
{
    Monitor::recordEvent("Allocate memory");
//    printf("total : %d\n", simParams.num_total_particle);
//    printf("sph: %d\n", simParams.num_sph_particle);
//    printf("pd: %d\n", simParams.num_pd_particle);

    // all particles
    size_t sizeREAL = sizeof(real_t) * simParams.num_sph_particle;
    size_t sizeInt = sizeof(int) * simParams.num_sph_particle;

    sizeMap[SPH_TIMESTEP] = sizeInt;
    typeMap[SPH_TIMESTEP] = INT32_TYPE;

    sizeMap[SPH_VALIDITY] = sizeInt;
    typeMap[SPH_VALIDITY] = INT32_TYPE;

    sizeMap[SPH_ACTIVITY] = sizeInt;
    typeMap[SPH_ACTIVITY] = INT32_TYPE;

    sizeMap[SPH_POSITION] = sizeREAL * 4;
    typeMap[SPH_POSITION] = REAL4_TYPE;

    sizeMap[SPH_VELOCITY] = sizeREAL * 4;
    typeMap[SPH_VELOCITY] = REAL4_TYPE;

    sizeMap[SPH_FORCE] = sizeREAL * 4;
    typeMap[SPH_FORCE] = REAL4_TYPE;

    sizeMap[SPH_SORTED_POSITION] = sizeREAL * 4;
    typeMap[SPH_SORTED_POSITION] = REAL4_TYPE;

    sizeMap[SPH_SORTED_VELOCITY] = sizeREAL * 4;
    typeMap[SPH_SORTED_VELOCITY] = REAL4_TYPE;

    sizeMap[SPH_SORTED_DENSITY] = sizeREAL;
    typeMap[SPH_SORTED_DENSITY] = REAL_TYPE;

    sizeMap[SPH_SORTED_DENSITY_NORMALIZED] = sizeREAL;
    typeMap[SPH_SORTED_DENSITY_NORMALIZED] = REAL_TYPE;

    sizeMap[SPH_SORTED_PRESSURE] = sizeREAL;
    typeMap[SPH_SORTED_PRESSURE] = REAL_TYPE;

    sizeREAL = sizeof(real_t) * (2 * (simParams.boundaryPlaneSize + 2) *
                                 (simParams.boundaryPlaneSize + 2)
                                 + 4 * simParams.boundaryPlaneSize *
                                 (simParams.boundaryPlaneSize + 1));

    sizeMap[SPH_BOUNDARY_POSITION] = sizeREAL * 4;
    typeMap[SPH_BOUNDARY_POSITION] = REAL4_TYPE;

    // Peridynamics particles only
    sizeREAL = sizeof(real_t) * simParams.num_pd_particle;
    sizeInt = sizeof(int) * simParams.num_pd_particle;

    sizeMap[PD_ACTIVITY] = sizeInt;
    typeMap[PD_ACTIVITY] = INT32_TYPE;

    sizeMap[PD_POSITION] = sizeREAL * 4;
    typeMap[PD_POSITION] = REAL4_TYPE;

    sizeMap[PD_POSITION_BACKUP] = sizeREAL * 4;
    typeMap[PD_POSITION_BACKUP] = REAL4_TYPE;

    sizeMap[PD_VELOCITY] = sizeREAL * 4;
    typeMap[PD_VELOCITY] = REAL4_TYPE;

    sizeMap[PD_VELOCITY_BACKUP] = sizeREAL * 4;
    typeMap[PD_VELOCITY_BACKUP] = REAL4_TYPE;

    sizeMap[PD_FORCE] = sizeREAL * 4;
    typeMap[PD_FORCE] = REAL4_TYPE;

    sizeMap[PD_SORTED_POSITION] = sizeREAL * 4;
    typeMap[PD_SORTED_POSITION] = REAL4_TYPE;

    sizeMap[PD_SORTED_VELOCITY] = sizeREAL * 4;
    typeMap[PD_SORTED_VELOCITY] = REAL4_TYPE;

    sizeMap[PD_ORIGINAL_POSITION] = sizeREAL * 4;
    typeMap[PD_ORIGINAL_POSITION] = REAL4_TYPE;

    sizeMap[PD_ORIGINAL_STRETCH] = sizeREAL;
    typeMap[PD_ORIGINAL_STRETCH] = REAL4_TYPE;

    sizeMap[PD_ORIGINAL_BOND_LIST_TOP] = sizeInt;
    typeMap[PD_ORIGINAL_BOND_LIST_TOP] = INT32_TYPE;

    sizeMap[PD_STRETCH] = sizeREAL;
    typeMap[PD_STRETCH] = REAL_TYPE;

    sizeMap[PD_NEW_STRETCH] = sizeREAL;
    typeMap[PD_NEW_STRETCH] = REAL_TYPE;

    sizeMap[PD_BOND_LIST_TOP] = sizeInt;
    typeMap[PD_BOND_LIST_TOP] = INT32_TYPE;

    sizeMap[PD_BOND_LIST_TOP_BACKUP] = sizeInt;
    typeMap[PD_BOND_LIST_TOP_BACKUP] = INT32_TYPE;

    sizeMap[PD_BOND_LIST] = sizeInt * MAX_PD_BOND_COUNT;
    typeMap[PD_BOND_LIST] = INT32_TYPE;

    sizeMap[PD_CLIST] = sizeof(Clist) * simParams.num_clists;
    typeMap[PD_CLIST] = STRUCT;



    sizeMap[PD_SYSTEM_MATRIX] = simParams.num_pd_particle * MAX_PD_BOND_COUNT * SIZE_MAT_3X3;
    typeMap[PD_SYSTEM_MATRIX] = STRUCT;

    sizeMap[PD_SYSTEM_VECTOR] = sizeREAL * 4;
    typeMap[PD_SYSTEM_VECTOR] = REAL4_TYPE;

    sizeMap[PD_SYSTEM_SOLUTION] = sizeREAL * 4;
    typeMap[PD_SYSTEM_SOLUTION] = REAL4_TYPE;

    sizeMap[PD_HAS_BROKEN_BOND] = sizeof(int);
    typeMap[PD_HAS_BROKEN_BOND] = INT32_TYPE;

    // grid cells
    sizeInt = sizeof(int) * simParams.num_cells;
    sizeMap[CELL_PARTICLE_TYPE] = sizeInt;
    typeMap[CELL_PARTICLE_TYPE] = INT32_TYPE;



    sizeInt = sizeof(int) * simParams.num_sph_particle;
    sizeMap[SPH_PARTICLE_TO_CELL_HASH] = sizeInt;
    typeMap[SPH_PARTICLE_TO_CELL_HASH] = INT32_TYPE;

    sizeMap[SPH_PARTICLE_UNSORTED_INDEX] = sizeInt;
    typeMap[SPH_PARTICLE_UNSORTED_INDEX] = INT32_TYPE;

    sizeInt = sizeof(int) * simParams.num_cells;
    sizeMap[SPH_CELL_START_INDEX] = sizeInt;
    typeMap[SPH_CELL_START_INDEX] = INT32_TYPE;

    sizeMap[SPH_CELL_END_INDEX] = sizeInt;
    typeMap[SPH_CELL_END_INDEX] = INT32_TYPE;



    sizeInt = sizeof(int) * simParams.num_pd_particle;
    sizeMap[PD_PARTICLE_TO_CELL_HASH] = sizeInt;
    typeMap[PD_PARTICLE_TO_CELL_HASH] = INT32_TYPE;

    sizeMap[PD_PARTICLE_UNSORTED_INDEX] = sizeInt;
    typeMap[PD_PARTICLE_UNSORTED_INDEX] = INT32_TYPE;

    sizeInt = sizeof(int) * simParams.num_cells;
    sizeMap[PD_CELL_START_INDEX] = sizeInt;
    typeMap[PD_CELL_START_INDEX] = INT32_TYPE;

    sizeMap[PD_CELL_END_INDEX] = sizeInt;
    typeMap[PD_CELL_END_INDEX] = INT32_TYPE;


    TRUE_OR_DIE(sizeMap.size() == NUM_VARIABLES,
                "Ohh, you've omitted to initialize some variables....");
    TRUE_OR_DIE(typeMap.size() == NUM_VARIABLES,
                "Ohh, you've omitted to initialize some variables....");


    // map variable to string
    variable2NameMap[SPH_TIMESTEP] = "SPH_TIMESTEP";
    variable2NameMap[SPH_VALIDITY] = "SPH_VALIDITY";
    variable2NameMap[SPH_ACTIVITY] = "SPH_ACTIVITY";

    variable2NameMap[SPH_POSITION] = "SPH_POSITION";
    variable2NameMap[SPH_VELOCITY] = "SPH_VELOCITY";
    variable2NameMap[SPH_FORCE] = "SPH_FORCE";
    variable2NameMap[SPH_SORTED_POSITION] = "SPH_SORTED_POSITION";
    variable2NameMap[SPH_SORTED_VELOCITY] = "SPH_SORTED_VELOCITY";
    variable2NameMap[SPH_SORTED_DENSITY] = "SPH_SORTED_DENSITY";
    variable2NameMap[SPH_SORTED_DENSITY_NORMALIZED] = "SPH_SORTED_NORMALIZED_DENSITY";
    variable2NameMap[SPH_SORTED_PRESSURE] = "SPH_SORTED_PRESSURE";

    variable2NameMap[PD_ACTIVITY] = "PD_ACTIVITY";
    variable2NameMap[PD_POSITION] = "PD_POSITION";
    variable2NameMap[PD_VELOCITY] = "PD_VELOCITY";
    variable2NameMap[PD_FORCE] = "PD_FORCE";
    variable2NameMap[PD_SORTED_POSITION] = "PD_SORTED_POSITION";
    variable2NameMap[PD_SORTED_VELOCITY] = "PD_SORTED_VELOCITY";
    variable2NameMap[PD_ORIGINAL_POSITION] = "PD_ORIGINAL_POSITION";
    variable2NameMap[PD_ORIGINAL_STRETCH] = "PD_ORIGINAL_STRETCH";
    variable2NameMap[PD_ORIGINAL_BOND_LIST_TOP] = "PD_ORIGINAL_BOND_COUNT";
    variable2NameMap[PD_STRETCH] = "PD_STRETCH";
    variable2NameMap[PD_BOND_LIST] = "PD_BOND_LIST";
    variable2NameMap[PD_BOND_LIST_TOP] = "PD_BOND_LIST_TOP";
    variable2NameMap[PD_BOND_LIST_TOP_BACKUP] = "PD_BOND_LIST_TOP_BACKUP";
    variable2NameMap[PD_CLIST] = "PD_NEIGHBOR_LIST";

    variable2NameMap[CELL_PARTICLE_TYPE] = "CELL_MIXED_PARTICLE";


    // allocation memory
    allocateHostMemory();
    allocateDeviceMemory();
}

//------------------------------------------------------------------------------------------
MemoryManager::~MemoryManager()
{
    for(std::map<Variables, void*>::iterator ptr = hostPointerMap.begin();
        ptr != hostPointerMap.end(); ++ptr)
    {
        delete[] (ptr->second);
    }

    for(std::map<Variables, void*>::iterator ptr = devicePointerMap.begin();
        ptr != devicePointerMap.end(); ++ptr)
    {
        freeDeviceArray(ptr->second);
    }

}

//------------------------------------------------------------------------------------------
void MemoryManager::uploadToDevice(MemoryManager::Variables _variable, bool _scale)
{
    void* source = hostPointerMap[_variable];
    void* dest = devicePointerMap[_variable];
    size_t size = sizeMap[_variable];

    if(_scale)
    {
        if(_variable == SPH_POSITION ||
           _variable == PD_POSITION ||
           _variable == SPH_SORTED_POSITION ||
           _variable == PD_SORTED_POSITION ||
           _variable == PD_ORIGINAL_POSITION ||
           _variable == SPH_BOUNDARY_POSITION)
        {
            real4_t* parPos = (real4_t*) source;
            int numElements = sizeMap[_variable] / (4 * sizeof(real_t));

            scaleParticle(parPos, numElements, 1.0 / simParams_.scaleFactor);
        }
    }

    checkCudaErrors(cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice));
}

//------------------------------------------------------------------------------------------
void MemoryManager::uploadAllArrayToDevice(bool _scale)
{
    for(int i = 0; i < NUM_VARIABLES; ++i)
    {
        Variables variable = static_cast<Variables>(i);
        uploadToDevice(variable, _scale);
    }

}

//------------------------------------------------------------------------------------------
void MemoryManager::downloadFromDevice(MemoryManager::Variables _variable)
{
    void* source = devicePointerMap[_variable];
    void* dest = hostPointerMap[_variable];
    size_t size = sizeMap[_variable];

    checkCudaErrors(cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost));

    // scale position, if needed
//    if(_scale)
    {
        if(_variable == SPH_POSITION ||
           _variable == PD_POSITION ||
           _variable == SPH_SORTED_POSITION ||
           _variable == PD_SORTED_POSITION ||
           _variable == PD_ORIGINAL_POSITION ||
           _variable == SPH_BOUNDARY_POSITION)
        {
            real4_t* parPos = (real4_t*) dest;
            int numElements = sizeMap[_variable] / (4 * sizeof(real_t));

            scaleParticle(parPos, numElements, simParams_.scaleFactor);
        }
    }
}

//------------------------------------------------------------------------------------------
void MemoryManager::downloadAllArrayFromDevice()
{
    for(int i = 0; i < NUM_VARIABLES; ++i)
    {
        Variables variable = static_cast<Variables>(i);
        downloadFromDevice(variable);
    }
}

//------------------------------------------------------------------------------------------
void MemoryManager::printArray(MemoryManager::Variables _variable, int _size)
{
    // download first
    downloadFromDevice(_variable);

    // then print
    printHostArray(_variable, _size);

}

//------------------------------------------------------------------------------------------
void MemoryManager::printHostArray(MemoryManager::Variables _variable, int _size)
{
    VariableTypes type = typeMap[_variable];
    int size = _size;

    std::cout << "==================== " << getVariableName(_variable) <<
              " ====================" << std::endl;

    switch(type)
    {
    case REAL_TYPE:
    {
        if(size == 0)
        {
            size = sizeMap[_variable] / sizeof(real_t);
        }

        real_t* data = (real_t*)hostPointerMap[_variable];

        for(int i = 0; i < size; ++i)
        {
            std::cout << "[" << i << "] " << std::scientific << data[i] << std::endl;
        }
    }
    break;

    case REAL4_TYPE:
    {
        if(size == 0)
        {
            size = sizeMap[_variable] / sizeof(real4_t);
        }

        real4_t* data = (real4_t*)hostPointerMap[_variable];

        for(int i = 0; i < size; ++i)
        {
            std::cout << "[" << i << "] " << std::scientific << data[i].x << ", " << data[i].y << ", "
                      << data[i].z <<
                      std::endl;
        }
    }
    break;

    case INT32_TYPE:
    {
        if(size == 0)
        {
            size = sizeMap[_variable] / sizeof(int32_t);
        }

        int* data = (int*) hostPointerMap[_variable];

        for(int i = 0; i < size; ++i)
        {
            std::cout << "[" << i << "] " << data[i] << std::endl;
        }
    }
    break;

    case STRUCT:
    {
        if(size == 0)
        {
            size = sizeMap[_variable] / sizeof(Clist);
        }

        Clist* data = (Clist*) hostPointerMap[_variable];

        for(int i = 0; i < size; ++i)
        {
            std::cout << "[" << i << "] (" << data[i].plist_top +  1 << ") ";

            for(int j = 0; j < data[i].plist_top + 1; ++j)
            {
                std::cout << data[i].plist[j] << ", ";
            }

            std::cout << std::endl;
        }
    }
    break;
    }
}

//------------------------------------------------------------------------------------------
void MemoryManager::printPositiveIntegerArray(MemoryManager::Variables _variable,
                                              int _size)
{

    // then print
    VariableTypes type = typeMap[_variable];

    if(type != INT32_TYPE)
    {
        return;
    }

    downloadFromDevice(_variable);

    int size = _size;

    std::cout << "==================== " << getVariableName(_variable) <<
              " ====================" << std::endl;

    if(size == 0)
    {
        size = sizeMap[_variable] / sizeof(int);
    }

    int* data = (int*) hostPointerMap[_variable];

    for(int i = 0; i < size; ++i)
    {
        if(data[i] > 0)
        {
            std::cout << "[" << i << "] " << data[i] << std::endl;
        }
    }

}

//------------------------------------------------------------------------------------------
void* MemoryManager::getHostPointer(MemoryManager::Variables _variable)
{
    return hostPointerMap[_variable];
}

//------------------------------------------------------------------------------------------
void* MemoryManager::getDevicePointer(MemoryManager::Variables _variable)
{
    return devicePointerMap[_variable];
}

//------------------------------------------------------------------------------------------
size_t MemoryManager::getArraySize(MemoryManager::Variables _variable)
{
    return sizeMap[_variable];
}

//------------------------------------------------------------------------------------------
char* MemoryManager::getVariableName(MemoryManager::Variables _variable)
{
    return variable2NameMap[_variable];
}

//------------------------------------------------------------------------------------------
void MemoryManager::countMemory()
{
    size_t totalMemory = 0;

    for(std::map<Variables, size_t>::iterator ptr = sizeMap.begin();
        ptr != sizeMap.end(); ++ptr)
    {
        size_t size = ptr->second;

        if(size == 0)
        {
            continue;
        }

        totalMemory += size;
    }

    std::cout << Monitor::PADDING << "Total memory allocation: " << totalMemory / 1048576 <<
              "(MB)" << std::endl;
}

//------------------------------------------------------------------------------------------
void MemoryManager::backupBondListTopIndex()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_BOND_LIST_TOP_BACKUP],
                               devicePointerMap[PD_BOND_LIST_TOP],
                               sizeMap[PD_BOND_LIST_TOP], cudaMemcpyDeviceToDevice));

}

//------------------------------------------------------------------------------------------
void MemoryManager::restoreBondListTopIndex()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_BOND_LIST_TOP],
                               devicePointerMap[PD_BOND_LIST_TOP_BACKUP],
                               sizeMap[PD_BOND_LIST_TOP], cudaMemcpyDeviceToDevice));

}

//------------------------------------------------------------------------------------------
void MemoryManager::backupPDPosition()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_POSITION_BACKUP],
                               devicePointerMap[PD_POSITION],
                               sizeMap[PD_POSITION], cudaMemcpyDeviceToDevice));
}

//------------------------------------------------------------------------------------------
void MemoryManager::restorePDPosition()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_POSITION],
                               devicePointerMap[PD_POSITION_BACKUP],
                               sizeMap[PD_POSITION], cudaMemcpyDeviceToDevice));
}

//------------------------------------------------------------------------------------------
void MemoryManager::backupPDVelocity()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_VELOCITY_BACKUP],
                               devicePointerMap[PD_VELOCITY],
                               sizeMap[PD_VELOCITY], cudaMemcpyDeviceToDevice));
}

//------------------------------------------------------------------------------------------
void MemoryManager::restorePDVelocity()
{
    checkCudaErrors(cudaMemcpy(devicePointerMap[PD_VELOCITY],
                               devicePointerMap[PD_VELOCITY_BACKUP],
                               sizeMap[PD_VELOCITY], cudaMemcpyDeviceToDevice));
}

//------------------------------------------------------------------------------------------
void MemoryManager::transferData(MemoryManager::Variables dest_var,
                                 MemoryManager::Variables source_var)
{
    void* source = devicePointerMap[source_var];
    void* dest = devicePointerMap[dest_var];
    size_t size = sizeMap[source_var];

    checkCudaErrors(cudaMemcpy(dest, source, size, cudaMemcpyDeviceToDevice));

}

//------------------------------------------------------------------------------------------
void MemoryManager::allocateHostMemory()
{
    for(std::map<Variables, size_t>::iterator ptr = sizeMap.begin();
        ptr != sizeMap.end(); ++ptr)
    {
        Variables variable = ptr->first;
        size_t size = ptr->second;

        if(size == 0)
        {
            continue;
        }

        hostPointerMap[variable] = malloc(size);
    }

}

//------------------------------------------------------------------------------------------
void MemoryManager::allocateDeviceMemory()
{
    for(std::map<Variables, size_t>::iterator ptr = sizeMap.begin();
        ptr != sizeMap.end(); ++ptr)
    {
        Variables variable = ptr->first;
        size_t size = ptr->second;

        if(size == 0)
        {
            continue;
        }

        void* devPtr;
        allocateDeviceArray((void**)&devPtr, size);
        devicePointerMap[variable] = devPtr;
//        std::cout << "alloc dev " << variable << " size " << size << ", p " << devPtr <<
//                  std::endl;
    }
}

//------------------------------------------------------------------------------------------
void MemoryManager::scaleParticle(real4_t* _parPos, int _numParticles,
                                  real_t _scaleFactor)
{
    for(int i = 0; i < _numParticles; ++i)
    {
        real4_t pos = _parPos[i];

        pos.x *= _scaleFactor;
        pos.y *= _scaleFactor;
        pos.z *= _scaleFactor;
        _parPos[i] = pos;
    }
}

