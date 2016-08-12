//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <cassert>
#include <algorithm>
#include <cmath>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include <cusp/monitor.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/diagonal.h>

#include "utilities.h"
#include "kdtree.h"
#include "simulator.h"
#include "simulator.cuh"
#include "cg_solver.cuh"
#include "implicit_euler.cuh"
#include "newmark_beta.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

typedef typename cusp::array1d_view< thrust::device_ptr<int>   > DeviceIndexArrayView;
typedef typename cusp::array1d_view< thrust::device_ptr<real_t> > DeviceValueArrayView;
//------------------------------------------------------------------------------------------
Simulator::Simulator(int deviceID,
    SimulationParameters& simParams,
    RunningParameters& runningParams):
  hostSimParams_(simParams),
  runningParams_(runningParams),
  simMemory_(simParams),
  dataIO_(simParams, runningParams, simMemory_),
  current_step_(0),
  current_substep_(0),
  CG_convergent_(true)
{
  // Set the compute device
  checkCudaErrors(cudaSetDevice(deviceID));

  mapDeviceMemory();

  simMemory_.countMemory();
  Monitor::recordEvent("Calculate occupancy for kernel execution");
  calculateKernelOccupancy();
}

//------------------------------------------------------------------------------------------
void Simulator::makeReady()
{
  if(hostSimParams_.num_pd_particle > 0)
  {
    initPeridynamicsBonds();
  }



  Monitor::recordEvent("Upload all data to device memory");
  simMemory_.uploadAllArrayToDevice(false);

  Monitor::recordEvent("Set initial value for device memory");
  updateSimParamsToDevice(); // upload to use the rest density

  if(hostSimParams_.num_pd_particle > 0)
  {
    initPDSolutions <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(
        pd_system_solution, hostSimParams_.num_pd_particle);
    getLastCudaError("Kernel execution failed: initPDSolutions");
  }

  if(hostSimParams_.num_sph_particle > 0)
  {
    initSPHParticleData <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>
      (sph_force,
       sph_sorted_density,
       sph_sorted_density_normalized,
       sph_sorted_pressure,
       sph_validity,
       sph_timestep);
    getLastCudaError("Kernel execution failed: initSPHParticleDensityPressure");
  }
}

//------------------------------------------------------------------------------------------
/********** Round a / b to nearest higher integer value **********/
inline uint iDivUp(uint a, uint b)
{
  if(a == 0 || b == 0)
  {
    return 0;
  }
  else
  {
    return (a % b != 0) ? (a / b + 1) : (a / b);
  }
}

void Simulator::calculateKernelOccupancy()
{
  // kernel size for sph particles only
  kernelSPH_.nthreads = min(hostSimParams_.num_sph_particle, DEFAULT_BLOCK_SIZE);;
  kernelSPH_.nblocks = iDivUp(hostSimParams_.num_sph_particle, kernelSPH_.nthreads);

  std::cout << Monitor::PADDING << "SPH particles: numThreads/numBlocks = " <<
    kernelSPH_.nthreads << "/" << kernelSPH_.nblocks << std::endl;

  // kernel size for peridynamics particles only
  kernelPD_.nthreads = min(hostSimParams_.num_pd_particle, DEFAULT_BLOCK_SIZE);
  kernelPD_.nblocks = iDivUp(hostSimParams_.num_pd_particle, kernelPD_.nthreads);

  std::cout << Monitor::PADDING << "Peridynamics particles: numThreads/numBlocks = " <<
    kernelPD_.nthreads << "/" << kernelPD_.nblocks << std::endl;

  // kernel size for grid cell
  kernelGrid_.nthreads = min(hostSimParams_.num_cells, DEFAULT_BLOCK_SIZE);
  kernelGrid_.nblocks = iDivUp(hostSimParams_.num_cells, kernelGrid_.nthreads);

  std::cout << Monitor::PADDING << "Grid cell: numThreads/numBlocks = " <<
    kernelGrid_.nthreads << "/" << kernelGrid_.nblocks << std::endl;

  assert((kernelSPH_.nblocks > 0 && kernelSPH_.nthreads > 0) ||
      (kernelPD_.nblocks > 0 && kernelPD_.nthreads > 0) );
}


//------------------------------------------------------------------------------------------
void Simulator::updateSimParamsToDevice()
{
  checkCudaErrors(cudaMemcpyToSymbol(simParams, &hostSimParams_,
        sizeof(SimulationParameters)));
}

//------------------------------------------------------------------------------------------
bool Simulator::finished()
{
  return ((current_step_ > runningParams_.final_step) || !CG_convergent_);
}

//------------------------------------------------------------------------------------------
bool Simulator::CGConvergence()
{
  return CG_convergent_;
}

//------------------------------------------------------------------------------------------
int Simulator::get_current_step() const
{
  return current_step_;
}

//------------------------------------------------------------------------------------------
int* Simulator::get_sph_activity()
{
  return (int*)simMemory_.getHostPointer(MemoryManager::SPH_ACTIVITY);
}

//------------------------------------------------------------------------------------------
real4_t* Simulator::get_sph_position()
{
  return (real4_t*)simMemory_.getHostPointer(MemoryManager::SPH_POSITION);
}

//------------------------------------------------------------------------------------------
real4_t* Simulator::get_sph_velocity()
{
  return (real4_t*)simMemory_.getHostPointer(MemoryManager::SPH_VELOCITY);
}

//------------------------------------------------------------------------------------------
real4_t* Simulator::get_sph_boundary_pos()
{
  return (real4_t*)simMemory_.getHostPointer(MemoryManager::SPH_BOUNDARY_POSITION);
}

//------------------------------------------------------------------------------------------
int* Simulator::get_pd_activity()
{
  return (int*)simMemory_.getHostPointer(MemoryManager::PD_ACTIVITY);
}

//------------------------------------------------------------------------------------------
real4_t* Simulator::get_pd_position()
{
  return (real4_t*)simMemory_.getHostPointer(MemoryManager::PD_POSITION);
}

//------------------------------------------------------------------------------------------
real4_t* Simulator::get_pd_velocity()
{
  return (real4_t*)simMemory_.getHostPointer(MemoryManager::PD_VELOCITY);
}

//------------------------------------------------------------------------------------------
DataIO& Simulator::dataIO()
{
  return dataIO_;
}

//------------------------------------------------------------------------------------------
void Simulator::integrateSPH()
{
  if(hostSimParams_.num_sph_particle <= 0)
  {
    return;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // update particle velocity and position
  monitor_.startTimer();

  updateSPHVelocity <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      sph_timestep,
      sph_velocity,
      sph_force);
  getLastCudaError("Kernel execution failed: updateSPHVelocity");

  updateSPHPosition <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      sph_validity,
      sph_timestep,
      sph_position,
      sph_velocity);
  getLastCudaError("Kernel execution failed: updateSPHPosition");



  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::UPDATE_VELOCITY_POSITION);

  calculateSPHActivity();
  collectParticles();
  limitSPHTimestep();
  calculateSPHDensityPressure();
  //simMemory.printArray(MemoryManager::SPH_SORTED_NORMALIZED_DENSITY);

  ////////////////////////////////////////////////////////////////////////////////
  // calculate forces for SPH particles
  monitor_.startTimer();
  calculateSPHParticleForces <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      sph_sorted_pos,
      sph_sorted_vel,
      sph_force,
      sph_sorted_density_normalized,
      sph_sorted_pressure,
      sph_unsorted_index,
      sph_cell_start_index,
      sph_cell_end_index);
  getLastCudaError("Kernel execution failed: calculateSPHParticleForces");
  //    simMemory.printArray(MemoryManager::FORCE);

  calculateSPHParticleForcesBoundary <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>
    (sph_activity,
     sph_sorted_pos,
     sph_sorted_vel,
     sph_force,
     sph_sorted_density_normalized,
     sph_sorted_pressure,
     sph_boundary_pos,
     sph_unsorted_index);
  getLastCudaError("Kernel execution failed: calculateSPHParticleForcesBoundary");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::CALCULATE_SPH_FORCE);




  ////////////////////////////////////////////////////////////////////////////////
  // update velocity again
  monitor_.startTimer();


  updateSPHVelocity <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      sph_timestep,
      sph_velocity,
      sph_force);
  getLastCudaError("Kernel execution failed: updateSPHVelocity");


  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::UPDATE_VELOCITY_POSITION);
  //    simMemory.downloadFromDevice(MemoryManager::VELOCITY);
  //    simMemory.printArray(MemoryManager::VELOCITY, 100);

}

//------------------------------------------------------------------------------------------
bool Simulator::integratePeridynamicsImplicitEuler()
{
  monitor_.startTimer();


  checkCudaErrors(cudaMemset(pd_system_matrix, 0,
        hostSimParams_.num_pd_particle * MAX_PD_BOND_COUNT * SIZE_MAT_3X3));
  checkCudaErrors(cudaMemset(pd_has_broken_bond, 0, sizeof(int)));

  calculatePDForceDerivative <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_position,
      pd_force,
      pd_original_pos,
      pd_original_stretch,
      pd_stretch,
      pd_new_stretch,
      pd_bond_list,
      pd_bond_list_top,
      pd_system_matrix,
      pd_has_broken_bond,
      hostSimParams_.num_pd_particle,
      hostSimParams_.pd_horizon,
      hostSimParams_.pd_particle_radius,
      hostSimParams_.pd_C_times_V,
      hostSimParams_.clockScale,
      hostSimParams_.pd_particle_mass
      );
  getLastCudaError("Kernel execution failed: calculatePDForceDerivative");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::CALCULATE_PERIDYNAMICS_FORCE);
  //        simMemory_.printArray(MemoryManager::PD_MATRIX_ELEMENTS);


  if(runningParams_.adaptive_integration)
  {
    // count the number of negative index
    int has_broken_bond;
    checkCudaErrors(cudaMemcpy(&has_broken_bond, pd_has_broken_bond, sizeof(int),
          cudaMemcpyDeviceToHost));


    if(has_broken_bond)
    {
      std::cout << Monitor::PADDING << Monitor::PADDING;
      std::cout << "Has broken bonds!" << std::endl;

      ////////////////////////////////////////////////////////////////////////////////
      // reduce time step by half and return, if the time step is > 1
      if(hostSimParams_.pd_time_step > TIMESTEP_PD_MIN)
      {

        hostSimParams_.pd_time_step /= 2;
        updateSimParamsToDevice();


        std::cout << Monitor::PADDING << Monitor::PADDING;
        std::cout << "Step back. Reduce time step to " << hostSimParams_.pd_time_step;
        std::cout << " times of time step base (10^-7)" << std::endl;

        ////////////////////////////////////////////////////////////////////////////////
        // restore memory
        simMemory_.restoreBondListTopIndex();
        simMemory_.restorePDPosition();
        simMemory_.restorePDVelocity();

        return false; // don't move this step
      }

      std::cout << Monitor::PADDING << Monitor::PADDING;
      std::cout << "Time step has reached minimum(" << TIMESTEP_PD_MIN << "). Advance." <<
        std::endl;
      current_substep_ = 1;
    }
    else
    {
      if(current_substep_ > 0)
      {
        ++current_substep_;

        printf("No fracture, substep %d...\n", current_substep_);
      }

      if(current_substep_ >= MAX_SUBSTEP)
      {
        // try to increase time step
        hostSimParams_.pd_time_step *= 2;
        updateSimParamsToDevice();

        std::cout << Monitor::PADDING << Monitor::PADDING;
        std::cout << "Try to increase time step twice to " << hostSimParams_.pd_time_step;
        std::cout << " times of time step base (10^-7)" << std::endl;

        current_substep_ = (hostSimParams_.pd_time_step >= TIMESTEP_PD) ? 0 : 1;
      }

    }

    ////////////////////////////////////////////////////////////////////////////////
    // backup data of this step
    simMemory_.backupBondListTopIndex();
    simMemory_.backupPDPosition();
    simMemory_.backupPDVelocity();
  }


  updatePDStretchNeighborList <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_stretch,
      pd_new_stretch,
      pd_clist);
  getLastCudaError("Kernel execution failed: updatePeriParticleStretch");

  //simMemory_.printArray(MemoryManager::PD_NEIGHBOR_LIST);

#if 0
  collectPDParticles <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pdActivity,
      pdPosition,
      pdNeighborList);
  getLastCudaError("Kernel execution failed: collectPDParticles");

  //        simMemory_.printArray(MemoryManager::PD_POSITION);
  //simMemory_.printArray(MemoryManager::PD_NEIGHBOR_LIST);

  collidePDParticles <<< kernelPD_.nblocks, kernelPD_.nthreads>>>
    (pdActivity,
     pdForce,
     pdPosition,
     pdNeighborList);
  getLastCudaError("Kernel execution failed: calculatePDCollisionForceDerivative");

#endif


  fillMatrixImplicitEuler <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_bond_list_top,
      pd_system_matrix);
  getLastCudaError("Kernel execution failed: fillMatrix");

  fillVectorBImplicitEuler <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_force,
      pd_velocity,
      pd_system_vector);
  getLastCudaError("Kernel execution failed: fillVectorB");



  solveLinearSystem();

  simMemory_.transferData(MemoryManager::PD_VELOCITY, MemoryManager::PD_SYSTEM_SOLUTION);

  updatePDPositionImplicitEuler <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_position,
      pd_velocity);
  getLastCudaError("Kernel execution failed: updatePDPosition");

  //        correctCollidedPDParticle <<< kernelPD_.nblocks, kernelPD_.nthreads>>>
  //        (pdActivity,
  //         pdVelocity,
  //         pdPosition,
  //         pdNeighborList);
  //        getLastCudaError("Kernel execution failed: correctCollidedPDParticle");



  return true;

}

//------------------------------------------------------------------------------------------
bool Simulator::integratePeridynamicsNewmarkBeta()
{

  monitor_.startTimer();


  checkCudaErrors(cudaMemset(pd_system_matrix, 0,
        hostSimParams_.num_pd_particle * MAX_PD_BOND_COUNT * SIZE_MAT_3X3));
  checkCudaErrors(cudaMemset(pd_has_broken_bond, 0, sizeof(int)));

  calculatePDForceDerivative <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_position,
      pd_force,
      pd_original_pos,
      pd_original_stretch,
      pd_stretch,
      pd_new_stretch,
      pd_bond_list,
      pd_bond_list_top,
      pd_system_matrix,
      pd_has_broken_bond,
      hostSimParams_.num_pd_particle,
      hostSimParams_.pd_horizon,
      hostSimParams_.pd_particle_radius,
      hostSimParams_.pd_C_times_V,
      hostSimParams_.clockScale,
      hostSimParams_.pd_particle_mass
      );
  getLastCudaError("Kernel execution failed: calculatePDForceDerivative");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::CALCULATE_PERIDYNAMICS_FORCE);
  //        simMemory_.printArray(MemoryManager::PD_MATRIX_ELEMENTS);


  // count the number of negative index
  int has_broken_bond;
  checkCudaErrors(cudaMemcpy(&has_broken_bond, pd_has_broken_bond, sizeof(int),
        cudaMemcpyDeviceToHost));


  if(has_broken_bond)
  {
    std::cout << Monitor::PADDING << Monitor::PADDING;
    std::cout << "Has broken bonds!" << std::endl;

    ////////////////////////////////////////////////////////////////////////////////
    // reduce time step by half and return, if the time step is > 1
    if(hostSimParams_.pd_time_step > TIMESTEP_PD_MIN)
    {

      hostSimParams_.pd_time_step /= 2;
      updateSimParamsToDevice();


      std::cout << Monitor::PADDING << Monitor::PADDING;
      std::cout << "Step back. Reduce time step to " << hostSimParams_.pd_time_step;
      std::cout << " times of time step base (10^-7)" << std::endl;

      ////////////////////////////////////////////////////////////////////////////////
      // restore memory
      simMemory_.restoreBondListTopIndex();
      simMemory_.restorePDPosition();
      simMemory_.restorePDVelocity();

      return false; // don't move this step
    }

    std::cout << Monitor::PADDING << Monitor::PADDING;
    std::cout << "Time step has reached minimum(" << TIMESTEP_PD_MIN << "). Advance." <<
      std::endl;
    current_substep_ = 1;
  }
  else
  {
    if(current_substep_ > 0)
    {
      ++current_substep_;

      printf("No fracture, substep %d...\n", current_substep_);
    }

    if(current_substep_ >= MAX_SUBSTEP)
    {
      // try to increase time step
      hostSimParams_.pd_time_step *= 2;
      updateSimParamsToDevice();

      std::cout << Monitor::PADDING << Monitor::PADDING;
      std::cout << "Try to increase time step twice to " << hostSimParams_.pd_time_step;
      std::cout << " times of time step base (10^-7)" << std::endl;

      current_substep_ = (hostSimParams_.pd_time_step >= TIMESTEP_PD) ? 0 : 1;
    }

  }

  ////////////////////////////////////////////////////////////////////////////////
  // backup data of this step
  simMemory_.backupBondListTopIndex();
  simMemory_.backupPDPosition();
  simMemory_.backupPDVelocity();


  updatePDStretchNeighborList <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_stretch,
      pd_new_stretch,
      pd_clist);
  getLastCudaError("Kernel execution failed: updatePeriParticleStretch");

  //simMemory_.printArray(MemoryManager::PD_NEIGHBOR_LIST);

#if 0
  collectPDParticles <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pdActivity,
      pdPosition,
      pdNeighborList);
  getLastCudaError("Kernel execution failed: collectPDParticles");

  //        simMemory_.printArray(MemoryManager::PD_POSITION);
  //simMemory_.printArray(MemoryManager::PD_NEIGHBOR_LIST);

  collidePDParticles <<< kernelPD_.nblocks, kernelPD_.nthreads>>>
    (pdActivity,
     pdForce,
     pdPosition,
     pdNeighborList);
  getLastCudaError("Kernel execution failed: calculatePDCollisionForceDerivative");

#endif

  fillMatrixNewmarkBeta <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_bond_list_top,
      pd_system_matrix);
  getLastCudaError("Kernel execution failed: fillMatrix");

  fillVectorBNewmarkBeta <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_force,
      pd_velocity,
      pd_system_vector);
  getLastCudaError("Kernel execution failed: fillVectorB");



  solveLinearSystem();

  updatePDVelocityNewmarkBeta <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_velocity,
      pd_system_solution);
  getLastCudaError("Kernel execution failed: updatePDVelocity");

  updatePDPositionNewmarkBeta <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_activity,
      pd_position,
      pd_velocity,
      pd_system_solution);
  getLastCudaError("Kernel execution failed: updatePDPosition");

  //        correctCollidedPDParticle <<< kernelPD_.nblocks, kernelPD_.nthreads>>>
  //        (pdActivity,
  //         pdVelocity,
  //         pdPosition,
  //         pdNeighborList);
  //        getLastCudaError("Kernel execution failed: correctCollidedPDParticle");

  return true;
}


//------------------------------------------------------------------------------------------
void Simulator::printDebugInfoLinearSystem()
{
  simMemory_.downloadFromDevice(MemoryManager::PD_SYSTEM_MATRIX);
  simMemory_.downloadFromDevice(MemoryManager::PD_SYSTEM_VECTOR);
  simMemory_.downloadFromDevice(MemoryManager::PD_BOND_LIST);
  simMemory_.downloadFromDevice(MemoryManager::PD_BOND_LIST_TOP);

  Mat3x3* sys_matrix = (Mat3x3*)simMemory_.getHostPointer(
      MemoryManager::PD_SYSTEM_MATRIX);
  real4_t* sys_vector = (real4_t*)simMemory_.getHostPointer(
      MemoryManager::PD_SYSTEM_VECTOR);

  int* bond_list = (int*) simMemory_.getHostPointer(MemoryManager::PD_BOND_LIST);
  int* bond_list_top = (int*) simMemory_.getHostPointer(MemoryManager::PD_BOND_LIST_TOP);


  static Mat3x3* matrix = new Mat3x3[hostSimParams_.num_pd_particle *
    hostSimParams_.num_pd_particle];

  for(int i = 0; i < hostSimParams_.num_pd_particle * hostSimParams_.num_pd_particle; ++i)
  {
    for(int j = 0; j < 3; ++j)
    {
      matrix[i].row[j] = MAKE_REAL3_FROM_REAL(0.0);
    }
  }

  for(int p = 0; p < hostSimParams_.num_pd_particle; ++p)
  {
    matrix[p * hostSimParams_.num_pd_particle + p] = sys_matrix[p];

    for(int bond = 0; bond <= bond_list_top[p]; ++bond)
    {
      int bond_index = bond * hostSimParams_.num_pd_particle + p;
      int q = bond_list[bond_index];
      matrix[p * hostSimParams_.num_pd_particle + q] = sys_matrix[bond_index +
        hostSimParams_.num_pd_particle];
    }

  }


  printf("=========================SYSTEM MATRIX=========================\n");

  for(int i = 0; i < hostSimParams_.num_pd_particle; ++i)
  {
    Mat3x3 element;

    for(int l = 0; l < 3; ++l)
    {
      printf("[");

      for(int j = 0; j < hostSimParams_.num_pd_particle - 1; ++j)
      {
        element = matrix[i * hostSimParams_.num_pd_particle + j];
        printf("%7.3e, %7.3e, %7.3e,   ", element.row[l].x, element.row[l].y, element.row[l].z);
      }

      element = matrix[i * hostSimParams_.num_pd_particle +
        hostSimParams_.num_pd_particle  - 1];
      printf("%7.3e, %7.3e, %7.3e]\n", element.row[l].x, element.row[l].y, element.row[l].z);

    }

  }


  printf("\n");
  printf("=========================SYSTEM VECTOR=========================\n");

  for(int i = 0; i < hostSimParams_.num_pd_particle; ++i)
  {
    printf("[%7.3e, %7.3e, %7.3e]\n", sys_vector[i].x, sys_vector[i].y, sys_vector[i].z);
  }

  printf("\n");
}

//------------------------------------------------------------------------------------------
void Simulator::solveLinearSystem()
{

  monitor_.startTimer();
#if PRINT_DEBUG
  {
    printDebugInfoLinearSystem();
  }
#endif

  static int size_vector = 4 * hostSimParams_.num_pd_particle;
  static PeridynamicsMatrix A(pd_bond_list, pd_bond_list_top, pd_system_matrix,
      size_vector,
      kernelPD_.nblocks, kernelPD_.nthreads, hostSimParams_.num_pd_particle);

  static thrust::device_ptr<real_t> wrapped_device_x((real_t*)pd_system_solution);
  static thrust::device_ptr<real_t> wrapped_device_b((real_t*)pd_system_vector);

  // use array1d_view to wrap the individual arrays
  typedef typename cusp::array1d_view< thrust::device_ptr<real_t> > DeviceValueArrayView;

  static DeviceValueArrayView x (wrapped_device_x, wrapped_device_x + size_vector);
  static DeviceValueArrayView b (wrapped_device_b, wrapped_device_b + size_vector);

#if PRINT_DEBUG
  cusp::verbose_monitor<real_t> cg_monitor(b, MAX_CG_ITERATION,
      CG_RELATIVE_TOLERANCE);
#else
  cusp::default_monitor<real_t> cg_monitor(b, MAX_CG_ITERATION,
      CG_RELATIVE_TOLERANCE);
#endif

  //    cusp::precond::scaled_bridson_ainv<real_t, cusp::device_memory> M(A);
  //        cusp::precond::diagonal<real_t, cusp::device_memory> M(A);
  cusp::identity_operator<real_t, cusp::device_memory> M(A.num_rows, A.num_rows);
  cusp::krylov::cg(A, x, b, cg_monitor, M);


  if (cg_monitor.converged())
  {
    std::cout << Monitor::PADDING << Monitor::PADDING;
    std::cout << "CG Solver converged to " << cg_monitor.tolerance() << " tolerance";
    std::cout << " after " << cg_monitor.iteration_count() << " iterations";
    std::cout << " (" << cg_monitor.residual_norm() << " final residual)" << std::endl;
  }
  else
  {
    std::cout << Monitor::PADDING << Monitor::PADDING;
    std::cout << "CG Solver reached iteration limit " << cg_monitor.iteration_limit() <<
      " before converging";
    std::cout << " to " << cg_monitor.tolerance() << " tolerance ";
    std::cout << " (" << cg_monitor.residual_norm() << " final residual)" << std::endl;

    CG_convergent_ = false;
    return;
  }



  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::SOLVE_LINEAR_EQUATION);


}

//------------------------------------------------------------------------------------------
inline real_t frand()
{
  return rand() / (real_t) RAND_MAX;
}

inline real_t getstretch(real_t s0)
{
  real_t sum;
  int i;
  sum = 0.0;

  // 12*U[0,1] - 6 gives N(0,1)
  // s0dev*N(0,1)+s0 gives N(s0,s0dev)
  for (i = 0; i < 12; ++i)
  {
    sum += frand();
  }

  sum -= 6.0;
  return ((sum * 0.01 + 1.0) * s0);
}


void Simulator::initPeridynamicsBonds()
{
  if(hostSimParams_.num_pd_particle <= 0)
  {
    std::cout << Monitor::PADDING << "No PD particle...." << std::endl;
    return;
  }

  real4_t* host_pd_pos = (real4_t*)simMemory_.getHostPointer(MemoryManager::PD_POSITION);
  real4_t* host_pd_opos = (real4_t*)simMemory_.getHostPointer(
      MemoryManager::PD_ORIGINAL_POSITION);
  real_t* host_pd_stretch = (real_t*)simMemory_.getHostPointer(MemoryManager::PD_STRETCH);
  real_t* host_pd_ostretch = (real_t*)simMemory_.getHostPointer(
      MemoryManager::PD_ORIGINAL_STRETCH);
  int* host_pd_bond_list = (int*)simMemory_.getHostPointer(MemoryManager::PD_BOND_LIST);
  int* host_pd_bond_list_top = (int*)simMemory_.getHostPointer(
      MemoryManager::PD_BOND_LIST_TOP);
  int* host_pd_obond_list_top = (int*)simMemory_.getHostPointer(
      MemoryManager::PD_ORIGINAL_BOND_LIST_TOP);

  /////////////////////////////////////////////////////////////////
  // init bonds
  int maxbonds = -1000000, minbonds = 1000000;
  KDNode* root;

  Point* point = new Point[hostSimParams_.num_pd_particle];

  srand(1973);

  for (size_t i = 0; i < hostSimParams_.num_pd_particle; ++i)
  {
    host_pd_opos[i] = host_pd_pos[i];
    host_pd_ostretch[i] = host_pd_stretch[i] = getstretch(runningParams_.pd_stretch_limit_s0);

    host_pd_bond_list_top[i] = -1;

    ////////////////////////////////////////////////////////////////////////////////
    point[i] = Point(host_pd_pos[i].x, host_pd_pos[i].y, host_pd_pos[i].z, i);
  }

  Point tiny(-1000000.0, -1000000.0, -1000000.0, -1);
  Point huge(1000000.0, 1000000.0, 1000000.0, -1);
  root = new KDNode(&point[0], hostSimParams_.num_pd_particle, tiny, huge);

  // Build initial tree.  The maximum number of points per leaf is a tough call.
  KDtree kd(hostSimParams_.num_pd_particle, root, 100, host_pd_bond_list,
      host_pd_bond_list_top,
      hostSimParams_.pd_horizon);
  kd.buildtree(root);

  for (size_t i = 0; i < hostSimParams_.num_pd_particle; ++i)
  {
    kd.find_neighbors(point[i], root);
  }


  std::cout << Monitor::PADDING << "Perform sorting for particle bonds..." << std::endl;
  int* tmpBonds = new int[MAX_PD_BOND_COUNT];
  int bondCount, k;
  int sumBonds = 0;

  for (size_t i = 0; i < hostSimParams_.num_pd_particle; ++i)
  {
    host_pd_obond_list_top[i] = host_pd_bond_list_top[i];

    // number of total bonds in the bond list of current particle, including itself
    bondCount = host_pd_bond_list_top[i] + 1;

    if ( bondCount > maxbonds)
    {
      maxbonds = bondCount;
    }

    if (bondCount < minbonds)
    {
      minbonds = bondCount;
    }

    sumBonds += (bondCount + 1); // count the particle i itself

    //        std::cout << i << " has num. bonds: " << bondCount << std::endl;

    for(size_t j = 0; j < bondCount; ++j)
    {
      k = host_pd_bond_list[j * hostSimParams_.num_pd_particle + i];
      //            std::cout << "Particles: " << i << ", bond: " << k << std::endl;

      if(i == k)
      {
        std::cout << "WARNINGGGGG: Particle " << i << " has self bond!!!" << std::endl;
      }

      tmpBonds[j] = k;
    }

    std::sort(tmpBonds, tmpBonds + bondCount);

    // std::cout << "After sort: " << std::endl;
    for(int j = 0; j < bondCount; ++j)
    {
      host_pd_bond_list[j * hostSimParams_.num_pd_particle + i] = tmpBonds[j];
      //            std::cout << "Particles: " << i << ", bond: " << tmpBonds[j] << std::endl;

    }

  }

  delete[] tmpBonds;

  //    sumBonds += hostSimParams_.num_pd_particle; // include the particle bonds to itself
  //    hostSimParams_.num_pd_sysmatrix_elements = sumBonds;
  hostSimParams_.pd_max_num_bonds = maxbonds;


  std::cout << Monitor::PADDING << "Number of Peridymanics particle bonds: " <<
    minbonds << " -> " << maxbonds << std::endl;
  std::cout << Monitor::PADDING << "Sum of bonds: " <<
    sumBonds << std::endl;
}

//------------------------------------------------------------------------------------------
void Simulator::calculateSPHActivity()
{
  ////////////////////////////////////////////////////////////////////////////////
  // find particles timestep and active particles
  monitor_.startTimer();

  initSPHParticleTimestep <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>> (sph_validity,
      sph_timestep);
  getLastCudaError("Kernel execution failed: calculateSPHTimestep");
  //    simMemory.printArray(MemoryManager::TIMESTEP);

  int numSteps = (hostSimParams_.num_pd_particle > 0) ? TIMESTEP_PD : TIMESTEP_SPH;

  updateSPHValidity <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_validity, numSteps);
  getLastCudaError("Kernel execution failed: updateValidity");

  findActiveSPHParticles <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      sph_validity);
  getLastCudaError("Kernel execution failed: findActiveParticles");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::CALCULATE_SPH_PARTICLE_TIMESTEP);
}

//------------------------------------------------------------------------------------------
void Simulator::collectParticles()
{
  ////////////////////////////////////////////////////////////////////////////////
  // collect particles
  monitor_.startTimer();


  calculateCellHash <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_cell_hash,
      sph_unsorted_index,
      sph_position,
      hostSimParams_.num_sph_particle);
  getLastCudaError("Kernel execution failed: calculateCellHash");


  // sort particles
  thrust::sort_by_key(thrust::device_ptr<int> (sph_cell_hash),
      thrust::device_ptr<int> (sph_cell_hash + hostSimParams_.num_sph_particle),
      thrust::device_ptr<int> (sph_unsorted_index));


  // set all cells to empty
  checkCudaErrors(cudaMemset(sph_cell_start_index,
        0xffffffff,
        hostSimParams_.num_cells * sizeof(int)));


  size_t smemSizeSPH = sizeof(int) * (kernelSPH_.nthreads + 1);
  collectSPHParticlesToCells <<< kernelSPH_.nblocks, kernelSPH_.nthreads, smemSizeSPH>>>
    (sph_cell_start_index,
     sph_cell_end_index,
     sph_cell_hash,
     sph_unsorted_index,
     sph_position,
     sph_velocity,
     sph_sorted_pos,
     sph_sorted_vel,
     hostSimParams_.num_sph_particle);
  getLastCudaError("Kernel execution failed: collectParticlesToCells");
  //        simMemory.printArray(MemoryManager::SPH_PARTICLE_TO_CELL_HASH);

  //    simMemory.printPositiveIntegerArray(MemoryManager::SPH_CELL_START_INDEX);
  //    simMemory.printPositiveIntegerArray(MemoryManager::SPH_CELL_END_INDEX);



  ////////////////////////////////////////////////////////////////////////////////
  // pd particles only
  if(hostSimParams_.num_pd_particle > 0)
  {
    calculateCellHash <<< kernelPD_.nblocks, kernelPD_.nthreads>>>(pd_cell_hash,
        pd_unsorted_index,
        pd_position,
        hostSimParams_.num_pd_particle);
    getLastCudaError("Kernel execution failed: calculateCellHash");


    thrust::sort_by_key(thrust::device_ptr<int> (pd_cell_hash),
        thrust::device_ptr<int> (pd_cell_hash + hostSimParams_.num_pd_particle),
        thrust::device_ptr<int> (pd_unsorted_index));


    checkCudaErrors(cudaMemset(pd_cell_start_index,
          0xffffffff,
          hostSimParams_.num_cells * sizeof(int)));
    size_t smemSizePD = sizeof(int) * (kernelPD_.nthreads + 1);

    collectPDParticlesToCells <<< kernelPD_.nblocks, kernelPD_.nthreads, smemSizePD>>>
      (pd_cell_start_index,
       pd_cell_end_index,
       pd_cell_hash,
       pd_unsorted_index,
       pd_position,
       pd_velocity,
       pd_sorted_pos,
       pd_sorted_vel,
       hostSimParams_.num_pd_particle);
    getLastCudaError("Kernel execution failed: collectParticlesToCells");

    //        simMemory.printArray(MemoryManager::P_PARTICLE_TO_CELL_HASH);
  }

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::COLLECT_PARTICLES);
}

//------------------------------------------------------------------------------------------
void Simulator::limitSPHTimestep()
{
  if(hostSimParams_.num_pd_particle <= 0)
  {
    return;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // determine the cell that has two type of particle
  monitor_.startTimer();

  findCellParticleType <<< kernelGrid_.nblocks, kernelGrid_.nthreads>>>(cell_type,
      sph_cell_start_index,
      pd_cell_start_index,
      sph_cell_end_index,
      pd_cell_end_index);
  getLastCudaError("Kernel execution failed: findCellMixedParticles");


  propagateMixedCell <<< kernelGrid_.nblocks, kernelGrid_.nthreads>>>(cell_type);
  getLastCudaError("Kernel execution failed: propagateMixedCell");

  findSemiActiveCell <<< kernelGrid_.nblocks, kernelGrid_.nthreads>>>(cell_type);
  getLastCudaError("Kernel execution failed: findSemiActiveCell");

  ////////////////////////////////////////////////////////////////////////////////
  // limit timestep for SPH particles that near Peridnyanics particles
  timestepLimiterCell <<< kernelGrid_.nblocks, kernelGrid_.nthreads>>>(cell_type,
      sph_activity,
      sph_timestep,
      sph_unsorted_index,
      sph_cell_start_index,
      sph_cell_end_index);
  getLastCudaError("Kernel execution failed: timestepLimiter");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::LIMIT_SPH_TIMESTEP);
}

//------------------------------------------------------------------------------------------
void Simulator::calculateSPHDensityPressure()
{
  ////////////////////////////////////////////////////////////////////////////////
  // calculate density
  monitor_.startTimer();

  calculateSPHParticleDensity <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      cell_type,
      sph_sorted_density,
      sph_sorted_pos,
      sph_unsorted_index,
      sph_cell_start_index,
      sph_cell_end_index);
  getLastCudaError("Kernel execution failed: density");

  //    simMemory.printArray(MemoryManager::SPH_ACTIVITY);
  //    simMemory.printArray(MemoryManager::SPH_SORTED_POSITION);
  //        simMemory.printArray(MemoryManager::SPH_SORTED_DENSITY);


  calculateSPHParticleDensityBoundary <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>
    (sph_activity,
     cell_type,
     sph_sorted_density,
     sph_sorted_pos,
     sph_boundary_pos,
     sph_unsorted_index);
  getLastCudaError("Kernel execution failed: densityBoundary");
  //    simMemory.printArray(MemoryManager::SPH_SORTED_DENSITY);


  normalizeDensity <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      cell_type,
      sph_sorted_density,
      sph_sorted_density_normalized,
      sph_sorted_pressure,
      sph_sorted_pos,
      sph_unsorted_index,
      sph_cell_start_index,
      sph_cell_end_index);
  getLastCudaError("Kernel execution failed: normalizeDensity");

  normalizeDensityBoundary <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(sph_activity,
      cell_type,
      sph_sorted_density,
      sph_sorted_density_normalized,
      sph_sorted_pressure,
      sph_sorted_pos,
      sph_boundary_pos,
      sph_unsorted_index);
  getLastCudaError("Kernel execution failed: normalizeDensityBoundary");

  checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
  checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
  monitor_.recordToEvent(Monitor::CALCULATE_SPH_DENSITY_PRESSURE);

}

//------------------------------------------------------------------------------------------
void Simulator::advance()
{
  ++current_step_;

  if(current_step_ <= PHASEIN)
  {
    hostSimParams_.clockScale = (real_t)current_step_ / (real_t)PHASEIN;
    updateSimParamsToDevice();
  }


  integrateSPH();


  if (hostSimParams_.num_pd_particle <= 0)
  {
    return;
  }


  if(runningParams_.integrator == IMPLICIT_EULER)
  {
    while(!integratePeridynamicsImplicitEuler());
  }
  else
  {
    while(!integratePeridynamicsNewmarkBeta());
  }

  if(hostSimParams_.num_sph_particle > 0 &&
      hostSimParams_.num_pd_particle > 0)
  {
    ////////////////////////////////////////////////////////////////////////////////
    // calculat interaction forces between SPH-PD
    monitor_.startTimer();
    collideDifferentParticles <<< kernelSPH_.nblocks, kernelSPH_.nthreads>>>(cell_type,
        sph_position,
        sph_sorted_pos,
        sph_sorted_vel,
        sph_velocity,
        sph_unsorted_index,
        pd_sorted_pos,
        pd_sorted_vel,
        pd_velocity,
        pd_unsorted_index,
        pd_cell_start_index,
        pd_cell_end_index);
    getLastCudaError("Kernel execution failed: collideDifferentParticles");

    checkCudaErrors(cudaEventRecord(monitor_.eStop, monitor_.stream));
    checkCudaErrors(cudaEventSynchronize(monitor_.eStop));
    monitor_.recordToEvent(Monitor::CALCULATE_SPH_PD_FORCE);
    //        printCell<<< cellNumBlocks, cellNumThreads>>>(cellMixedParticle);
  }




}

//------------------------------------------------------------------------------------------
void Simulator::printSimulationTime()
{
  monitor_.printSimulationTime();
  monitor_.resetTimer();
}

//------------------------------------------------------------------------------------------
void Simulator::mapDeviceMemory()
{
  sph_timestep = (int*)simMemory_.getDevicePointer(MemoryManager::SPH_TIMESTEP);
  sph_validity = (int*)simMemory_.getDevicePointer(MemoryManager::SPH_VALIDITY);
  sph_activity = (int*)simMemory_.getDevicePointer(MemoryManager::SPH_ACTIVITY);


  sph_position = (real4_t*)simMemory_.getDevicePointer(MemoryManager::SPH_POSITION);
  sph_velocity = (real4_t*)simMemory_.getDevicePointer(MemoryManager::SPH_VELOCITY);
  sph_force = (real4_t*)simMemory_.getDevicePointer(MemoryManager::SPH_FORCE);
  sph_sorted_pos = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::SPH_SORTED_POSITION);
  sph_sorted_vel = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::SPH_SORTED_VELOCITY);
  sph_sorted_density = (real_t*)simMemory_.getDevicePointer(
      MemoryManager::SPH_SORTED_DENSITY);
  sph_sorted_density_normalized = (real_t*) simMemory_.getDevicePointer(
      MemoryManager::SPH_SORTED_DENSITY_NORMALIZED);
  sph_sorted_pressure = (real_t*)simMemory_.getDevicePointer(
      MemoryManager::SPH_SORTED_PRESSURE);
  sph_boundary_pos = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::SPH_BOUNDARY_POSITION);

  pd_activity = (int*)simMemory_.getDevicePointer(MemoryManager::PD_ACTIVITY);
  pd_position = (real4_t*)simMemory_.getDevicePointer(MemoryManager::PD_POSITION);

  pd_velocity = (real4_t*)simMemory_.getDevicePointer(MemoryManager::PD_VELOCITY);
  pd_force = (real4_t*)simMemory_.getDevicePointer(MemoryManager::PD_FORCE);
  pd_sorted_pos = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_SORTED_POSITION);
  pd_sorted_vel = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_SORTED_VELOCITY);
  pd_original_pos = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_ORIGINAL_POSITION);
  pd_sorted_pos = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_SORTED_POSITION);
  pd_original_stretch = (real_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_ORIGINAL_STRETCH);
  pd_original_bond_list_top = (int*)simMemory_.getDevicePointer(
      MemoryManager::PD_ORIGINAL_BOND_LIST_TOP);
  pd_stretch = (real_t*)simMemory_.getDevicePointer(MemoryManager::PD_STRETCH);
  pd_new_stretch = (real_t*)simMemory_.getDevicePointer(MemoryManager::PD_NEW_STRETCH);
  pd_bond_list_top = (int*)simMemory_.getDevicePointer(MemoryManager::PD_BOND_LIST_TOP);
  pd_bond_list = (int*)simMemory_.getDevicePointer(MemoryManager::PD_BOND_LIST);
  pd_clist = (Clist*)simMemory_.getDevicePointer(MemoryManager::PD_CLIST);


  pd_system_matrix = (Mat3x3*)simMemory_.getDevicePointer(MemoryManager::PD_SYSTEM_MATRIX);
  pd_system_vector = (real4_t*)simMemory_.getDevicePointer(MemoryManager::PD_SYSTEM_VECTOR);
  pd_system_solution = (real4_t*)simMemory_.getDevicePointer(
      MemoryManager::PD_SYSTEM_SOLUTION);
  pd_has_broken_bond = (int*)simMemory_.getDevicePointer(MemoryManager::PD_HAS_BROKEN_BOND);

  cell_type = (int*)simMemory_.getDevicePointer(
      MemoryManager::CELL_PARTICLE_TYPE);



  sph_cell_hash = (int*)simMemory_.getDevicePointer(
      MemoryManager::SPH_PARTICLE_TO_CELL_HASH);
  sph_unsorted_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::SPH_PARTICLE_UNSORTED_INDEX);
  sph_cell_start_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::SPH_CELL_START_INDEX);
  sph_cell_end_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::SPH_CELL_END_INDEX);



  pd_cell_hash = (int*)simMemory_.getDevicePointer(
      MemoryManager::PD_PARTICLE_TO_CELL_HASH);
  pd_unsorted_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::PD_PARTICLE_UNSORTED_INDEX);
  pd_cell_start_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::PD_CELL_START_INDEX);
  pd_cell_end_index = (int*)simMemory_.getDevicePointer(
      MemoryManager::PD_CELL_END_INDEX);




}
