//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef SCENE_H
#define SCENE_H

#include <vector_types.h>

#include <core/definitions.h>
#include <core/parameters.h>
#include "mesh_query0.1/mesh_query.h"
#include "cyPoint.h"
#include <cstdlib>

#define JITTER 0.01

inline real_t frand()
{
    return rand() / (real_t) RAND_MAX;
}


class Scene
{
public:
    enum ParticleArrangement
    {
        REGULAR_GRID,
        JITTERED_GRID
    };

    Scene(SimulationParameters& simParams,
          RunningParameters& runningParams,
          ParticleArrangement _arrangement = Scene::REGULAR_GRID);

    void initSPHParticles(int* sph_activity, real4_t* sph_position, real4_t* sph_velocity);
    void initSPHBoundaryParticles(real4_t* sph_boundary_pos);
    void initPeridynamicsParticles(int* pd_activity, real4_t* pd_position,
                                   real4_t* pd_velocity);

protected:
    void fillParticles(real4_t* particles, int* grid3d, real_t* margin3,
                       real_t border, real_t spacing, real_t jitter, int num_particles,
                       bool position_correction);
    int fillParticlesToMesh(MeshObject* meshObject, cyPoint3f box_min,
                                       cyPoint3f box_max, real4_t* particles, int* grid3d,
                                       real_t spacing, real_t jitter, int max_num_particles);
    void fillTubeParticles(real4_t* particles, int tube_radius, real3_t base_center,
                           int* up_direction, real_t spacing, real_t jitter, int num_particles);
    void createSPHGrid(int* grid3d);
    void createPeridynamicsGrid(int* grid3d);
    void createPeridynamicsGrid(cyPoint3f box_min, cyPoint3f box_max, int* grid3d);

    void transformParticles(real4_t* particles, real4_t translation, real4_t rotation,
                            int num_particles);

    SimulationParameters& simParams_;
    RunningParameters& runningParams_;

    ParticleArrangement arrangement_;

    real4_t* pd_position_cache_;
};

#endif // SCENE_H
