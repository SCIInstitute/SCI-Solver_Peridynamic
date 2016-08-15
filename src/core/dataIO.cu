//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <sys/stat.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#if WIN32
#include <direct.h>
#else
#endif
#include "monitor.h"
#include "dataIO.h"

//------------------------------------------------------------------------------------------
DataIO::DataIO(SimulationParameters& simParams,
               RunningParameters& runningParams,
               MemoryManager& simMemory):
    simParams_(simParams),
    runningParams_(runningParams),
    simMemory_(simMemory),
    savedFrame(0),
    writtenSimulationParameter(false)
{
    savingPath = runningParams_.saving_path;

    for(size_t i = 0; i < MemoryManager::NUM_VARIABLES; ++i)
    {
        MemoryManager::Variables variable = static_cast<MemoryManager::Variables>(i);
        savingMap[variable] = 0;
    }


    savingMap[MemoryManager::SPH_POSITION] = 1;
    savingMap[MemoryManager::PD_POSITION] = 1;

    if(runningParams_.dump_velocity)
    {
        savingMap[MemoryManager::SPH_VELOCITY] = 1;
        savingMap[MemoryManager::PD_VELOCITY] = 1;
    }


    if(runningParams_.dump_activity)
    {
        savingMap[MemoryManager::SPH_ACTIVITY] = 1;
        savingMap[MemoryManager::PD_ACTIVITY] = 1;
    }

    if(runningParams_.dump_density)
    {
        savingMap[MemoryManager::SPH_SORTED_POSITION] = 1;
        savingMap[MemoryManager::SPH_SORTED_DENSITY_NORMALIZED] = 1;
    }

    if(runningParams_.dump_pd_bond_list_top)
    {
        savingMap[MemoryManager::PD_ORIGINAL_BOND_LIST_TOP] = 1;
        savingMap[MemoryManager::PD_BOND_LIST_TOP] = 1;
    }


    createOutputFolders();
}

//------------------------------------------------------------------------------------------
DataIO::~DataIO()
{
    delete[] savingPath;
}


//------------------------------------------------------------------------------------------
void DataIO::getDataAndWrite()
{
    for(size_t i = 0; i < MemoryManager::NUM_VARIABLES; ++i)
    {
        MemoryManager::Variables variable = static_cast<MemoryManager::Variables>(i);

        if(savingMap[variable] == 1)
        {
            getDataAndWrite(variable);
        }
    }
}
//------------------------------------------------------------------------------------------
void DataIO::getDataAndWrite(MemoryManager::Variables _variable)
{
    if(!writtenSimulationParameter)
    {
        writeSimulationParameter();
        writtenSimulationParameter = true;
    }

    if(simMemory_.getArraySize(_variable) == 0)
    {
        return;
    }

    std::cout << Monitor::PADDING << "Save data: " << simMemory_.getVariableName(
                  _variable) << std::endl;

    simMemory_.downloadFromDevice(_variable);

    FILE* fptr;
    char buff[512];
    sprintf(buff, "%s/DATA/%s/frame.%lu", savingPath, simMemory_.getVariableName(_variable),
            savedFrame);

    fptr = fopen(buff, "w");
    TRUE_OR_DIE(fptr, "Could not open file for write.");

    fwrite(simMemory_.getHostPointer(_variable), 1, simMemory_.getArraySize(_variable), fptr);
    fclose(fptr);
//std::cout << "save " << buff << ", size=" << simMemory_.getArraySize(_variable) << std::endl;
}

//------------------------------------------------------------------------------------------
void DataIO::saveState(size_t _timestep)
{
    // alway write simParam to disk, ask the params may change
    writeSimulationParameter();

    FILE* fptr;
    char buff[512];

    sprintf(buff, "%s/STATE/state.%lu", savingPath, _timestep);

    fptr = fopen(buff, "w");
    TRUE_OR_DIE(fptr, "Could not open file for write.");

    sprintf(buff, "Save state at timestep: %lu", _timestep);
    Monitor::recordEvent(buff);

    fwrite(&savedFrame, 1, sizeof(size_t), fptr);
    simMemory_.downloadAllArrayFromDevice();

    for(int i = 0; i < MemoryManager::NUM_VARIABLES; ++i)
    {
        MemoryManager::Variables variable = static_cast<MemoryManager::Variables>(i);
        fwrite(simMemory_.getHostPointer(variable), 1, simMemory_.getArraySize(variable), fptr);
    }

    fclose(fptr);
}

//------------------------------------------------------------------------------------------
void DataIO::loadState(size_t _timestep)
{
    FILE* fptr;
    char buff[512];

    /////////////////////////////////////////////////////////////////
    // load simulation parameters


    int bkNumSPHParticles = simParams_.num_sph_particle;
    int bkNumPDParticles = simParams_.num_pd_particle;
    int bkNumCells = simParams_.num_cells;

    readSimulationParameter(simParams_);

    TRUE_OR_DIE((bkNumSPHParticles == simParams_.num_sph_particle) &&
                (bkNumPDParticles == simParams_.num_pd_particle) &&
                (bkNumCells == simParams_.num_cells),
                "Inconsistent number of particles/cells with saved data.");

    // don't need to write simparam anymore
    writtenSimulationParameter = true;

    /////////////////////////////////////////////////////////////////
    // load memory

    sprintf(buff, "%s/STATE/state.%lu", savingPath, _timestep);
    fptr = fopen(buff, "r");
    TRUE_OR_DIE(fptr, "Could not open file for read.");

    sprintf(buff, "Read saved state at timestep: %lu", _timestep);
    Monitor::recordEvent(buff);

    fread(&savedFrame, 1, sizeof(size_t), fptr);

    for(int i = 0; i < MemoryManager::NUM_VARIABLES; ++i)
    {
        MemoryManager::Variables variable = static_cast<MemoryManager::Variables>(i);
        fread(simMemory_.getHostPointer(variable), 1, simMemory_.getArraySize(variable), fptr);
    }

    fclose(fptr);

    Monitor::recordEvent("Upload all data to device memory");

    simMemory_.uploadAllArrayToDevice(true);
}

//------------------------------------------------------------------------------------------
void DataIO::newFrame()
{
    ++savedFrame;
}

//------------------------------------------------------------------------------------------
void DataIO::createOutputFolders()
{
  struct stat info;
  Monitor::recordEvent("Prepare output folder");
  // create dirs that don't exist
  //main DATA
  auto datadir = std::string(savingPath) + "/DATA";
  if (stat(datadir.c_str(), &info) != 0 ||
    !(info.st_mode & S_IFDIR)) {
    mkdir(datadir.c_str());
  }
  //STATE
  datadir = std::string(savingPath) + "/DATA/STATE";
  if (stat(datadir.c_str(), &info) != 0 ||
    !(info.st_mode & S_IFDIR)) {
    mkdir(datadir.c_str());
  }
  //frames
  for (size_t i = 0; i < MemoryManager::NUM_VARIABLES; ++i) {
    MemoryManager::Variables variable = static_cast<MemoryManager::Variables>(i);
    if (savingMap[variable] == 1) {
      auto dir = std::string(savingPath) + "/DATA/" + simMemory_.getVariableName(variable);
      if (stat(dir.c_str(), &info) != 0 ||
        !(info.st_mode & S_IFDIR)) {
        mkdir(dir.c_str());
      }
    }
  }
}

//------------------------------------------------------------------------------------------
void DataIO::writeSimulationParameter()
{
    FILE* fptr;
    char buff[512];

    sprintf(buff, "%s/DATA/sim_info.dat", savingPath);

    fptr = fopen(buff, "w");
    TRUE_OR_DIE(fptr, "Could not open file for write.");

    fwrite((const void*)&simParams_, 1, sizeof(SimulationParameters),
           fptr);
    fclose(fptr);

    // write for visualization
    sprintf(buff, "%s/DATA/viz_info.dat", savingPath);

    std::ofstream outFile(buff, std::ofstream::out);
    TRUE_OR_DIE(outFile.is_open(), "Could not open parameter file.");

    outFile << "num_total_particle " << simParams_.num_total_particle << std::endl;
    outFile << "num_sph_particle " << simParams_.num_sph_particle << std::endl;
    outFile << "num_pd_particle " << simParams_.num_pd_particle << std::endl;

    outFile << "sph_kernel_coeff " << simParams_.sph_kernel_coeff << std::endl;
    outFile << "pd_kernel_coeff " << simParams_.pd_kernel_coeff << std::endl;
    outFile << "sph_particle_mass " << simParams_.sph_particle_mass << std::endl;
    outFile << "pd_particle_mass " << simParams_.pd_particle_mass << std::endl;
    outFile << "sph_sph_viscosity " << simParams_.sph_sph_viscosity << std::endl;
    outFile << "sph_boundary_viscosity " << simParams_.sph_boundary_viscosity << std::endl;
    outFile << "sph_pd_slip " << simParams_.sph_pd_slip << std::endl;

    outFile << "boundary_min_x " << simParams_.boundary_min_x << std::endl;
    outFile << "boundary_min_y " << simParams_.boundary_min_y << std::endl;
    outFile << "boundary_min_z " << simParams_.boundary_min_z << std::endl;
    outFile << "boundary_max_x " << simParams_.boundary_max_x << std::endl;
    outFile << "boundary_max_y " << simParams_.boundary_max_y << std::endl;
    outFile << "boundary_max_z " << simParams_.boundary_max_z << std::endl;

    outFile << "sph_particle_radius " << simParams_.sph_particle_radius << std::endl;
    outFile << "sph_rest_density " << simParams_.sph_rest_density << std::endl;
    outFile << "pd_particle_radius " << simParams_.pd_particle_radius << std::endl;
    outFile << "pd_horizon " << simParams_.pd_horizon << std::endl;
    outFile << "scaleFactor " << simParams_.scaleFactor << std::endl;



    outFile.close();
}


//------------------------------------------------------------------------------------------
void DataIO::readSimulationParameter(SimulationParameters& _simParams)
{
    FILE* fptr;
    char buff[512];

    sprintf(buff, "%s/DATA/sim_info.dat", savingPath);

    fptr = fopen(buff, "r");
    TRUE_OR_DIE(fptr, "Could not open file for read.");

    fread((void*)&_simParams, 1, sizeof(SimulationParameters), fptr);
    fclose(fptr);


}
