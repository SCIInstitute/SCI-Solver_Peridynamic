//------------------------------------------------------------------------------------------
//
//
// Created on: 2/1/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <iostream>
#include <cmath>
#include <core/helper_math.h>
#include "scene.h"

#include "core/cutil_math_ext.h"
#include "core/monitor.h"
#include "cyTriMesh.h"

//------------------------------------------------------------------------------------------
Scene::Scene(SimulationParameters& simParams,
             RunningParameters& runningParams,
             ParticleArrangement _arrangement):
    simParams_(simParams),
    runningParams_(runningParams),
    arrangement_(_arrangement)
{
    if(simParams_.num_pd_particle == 0)
    {
        return;
    }

    cyTriMesh triMesh;
    triMesh.LoadFromFileObj(runningParams_.obj_file);
    triMesh.ComputeBoundingBox();
    cyPoint3f box_min = triMesh.GetBoundMin();
    cyPoint3f box_max = triMesh.GetBoundMax();
    cyPoint3f diff = box_max - box_min;
    double maxDiff = fmaxf(fmaxf(fabs(diff.x), fabs(diff.y)), fabs(diff.z));
    double scale = fminf(fminf(simParams_.boundary_max_x - simParams_.boundary_min_x,
                               simParams_.boundary_max_y - simParams_.boundary_min_y),
                         simParams_.boundary_max_z - simParams_.boundary_min_z) * 0.9 / maxDiff;

    // how small the mesh object will be:
    scale *= 0.5f;
//    scale /= simParams_.scaleFactor;
//    scale = fmin(scale*0.9, 1.0);

    box_min.x *= scale;
    box_min.y *= scale;
    box_min.z *= scale;
    box_max.x *= scale;
    box_max.y *= scale;
    box_max.z *= scale;

    // translate the object if needed
    double shift_x = simParams_.boundary_min_x - box_min.x;
    double shift_y = simParams_.boundary_min_y - box_min.y;
    double shift_z = simParams_.boundary_min_z - box_min.z;

    box_min.x += shift_x;
    box_max.x += shift_x;

    box_min.y += shift_y;
    box_max.y += shift_y;

    box_min.z += shift_z;
    box_max.z += shift_z;


    std::cout << Monitor::PADDING << "Bounding box for mesh object: [" << box_min.x << ", " <<
              box_min.y << ", " << box_min.z << "] -> [" << box_max.x << ", " << box_max.y << ", "
              << box_max.z << "]" << std::endl;

    int grid3d[3];
    createPeridynamicsGrid(box_min, box_max, grid3d);

    int max_num_pd_particles = grid3d[0] * grid3d[1] * grid3d[2];

    double* vertices = new double[triMesh.NV() * 3];
    int* faces = new int[triMesh.NF() * 3];

    for(int i = 0; i < triMesh.NV(); ++i)
    {
        cyPoint3f vertex = triMesh.V(i);
        vertices[i * 3] = (double) vertex[0] * scale + shift_x;
        vertices[i * 3 + 1] = (double) vertex[1] * scale + shift_y;
        vertices[i * 3 + 2] = (double) vertex[2] * scale + shift_z;
    }

    for(int i = 0; i < triMesh.NF(); ++i)
    {
        cyTriMesh::cyTriFace face = triMesh.F(i);
        faces[i * 3] = (int) face.v[0];
        faces[i * 3 + 1] = (int) face.v[1];
        faces[i * 3 + 2] = (int) face.v[2];
    }

    MeshObject* meshObject = construct_mesh_object(triMesh.NV(), vertices, triMesh.NF(),
                                                   faces);
    pd_position_cache_ = new real4_t[max_num_pd_particles];

    real_t jitter;
    real_t margin;
    real_t spacing;

    if(arrangement_ == Scene::REGULAR_GRID)
    {
        jitter = 0.0f;
    }
    else
    {
        jitter = JITTER * simParams_.pd_particle_radius;
    }

    margin = simParams_.pd_particle_radius;
    spacing = 2 * simParams_.pd_particle_radius;


    simParams_.num_pd_particle = fillParticlesToMesh(meshObject, box_min, box_max,
                                                     pd_position_cache_, grid3d, spacing,
                                                     jitter, max_num_pd_particles);
    simParams_.num_total_particle = simParams_.num_sph_particle + simParams_.num_pd_particle;

    simParams_.num_clists = simParams_.num_pd_particle;

    while(simParams_.num_clists % 8 != 0)
    {
        simParams_.num_clists++;
    }

//    else
//    {
//        simParams_.num_clists = (int) (floor(simParams_.num_pd_particle / 8.0) + 1) * 8;
//    }

    std::cout << Monitor::PADDING << "Num. clist: " << simParams_.num_clists << std::endl;

}

//------------------------------------------------------------------------------------------
void Scene::initSPHParticles(int* sph_activity, real4_t* sph_position,
                             real4_t* sph_velocity)
{
    if(simParams_.num_sph_particle == 0)
    {
        return;
    }

    real_t jitter;
    real_t margin3d[3];
    real_t spacing, border;

    if(arrangement_ == Scene::REGULAR_GRID)
    {
        jitter = 0.0f;
    }
    else
    {
        jitter = JITTER * simParams_.sph_particle_radius;
    }

    margin3d[0] = simParams_.sph_particle_radius;
    margin3d[1] = simParams_.sph_particle_radius;
    margin3d[2] = simParams_.sph_particle_radius;
    spacing = 2 * simParams_.sph_particle_radius;
    border = simParams_.sph_particle_radius;

    int grid3d[3];

    createSPHGrid(grid3d);

    srand(1546);

    fillParticles(sph_position, grid3d, margin3d, border,
                  spacing, jitter,
                  simParams_.num_sph_particle, true);

    // set the activity and velocity
    for(int i = 0; i < simParams_.num_sph_particle; ++i)
    {
        sph_activity[i] = ACTIVE;
        sph_velocity[i] = MAKE_REAL4(0, 0, runningParams_.sph_initial_velocity, 0);

    }
}

//------------------------------------------------------------------------------------------
inline double kernel_poly6(const real_t t)
{
    double val = 0.0;

    if(t >= 1.0)
    {
        return val;
    }

    const double tmp =  1.0 - t;
    val = tmp * tmp * tmp;

    return val;
}
void Scene::initSPHBoundaryParticles(real4_t* sph_boundary_pos)
{
    if(simParams_.num_sph_particle == 0)
    {
        return;
    }

    int plane_size = simParams_.boundaryPlaneSize;

    real_t spacing = 2 * simParams_.sph_particle_radius;

    int index = 0;

    int num_plane_particles = (plane_size + 2) * (plane_size + 2);
    simParams_.boundaryPlaneBottomIndex = index;
    simParams_.boundaryPlaneBottomSize = num_plane_particles;
    simParams_.boundaryPlaneTopIndex = index + num_plane_particles;
    simParams_.boundaryPlaneTopSize = num_plane_particles;
    real_t px, py, pz;

    // The top and bottom planes have size (size+2)^2
    for (int x = 0; x < plane_size + 2; x++)
    {
        for (int z = 0; z < plane_size + 2; z++)
        {
            // Bottom plane
            px = spacing * x + simParams_.boundary_min_x - simParams_.sph_particle_radius;
            py = simParams_.boundary_min_y - simParams_.sph_particle_radius;
            pz = spacing * z + simParams_.boundary_min_z - simParams_.sph_particle_radius;

            sph_boundary_pos[index] = MAKE_REAL4(px, py, pz, 0.0f);

            // Top plane
            px = spacing * x + simParams_.boundary_min_x - simParams_.sph_particle_radius;
            py = simParams_.boundary_min_y + simParams_.sph_particle_radius;
            pz = spacing * z + simParams_.boundary_min_z - simParams_.sph_particle_radius;
            sph_boundary_pos[index + num_plane_particles] = MAKE_REAL4(px, py, pz, 0.0f);

            index++;
        }
    }

    index += num_plane_particles;

    num_plane_particles = (plane_size + 1) * plane_size;
    simParams_.boundaryPlaneFrontIndex = index;
    simParams_.boundaryPlaneFrontSize = num_plane_particles;
    simParams_.boundaryPlaneBackIndex = index + num_plane_particles;
    simParams_.boundaryPlaneBackSize = num_plane_particles;

    // Front and back plane have size (size+1)*size
    for (int x = 0; x < plane_size + 1; x++)
    {
        for (int y = 0; y < plane_size; y++)
        {
            // Front plane
            px = spacing * x + simParams_.boundary_min_x + simParams_.sph_particle_radius;
            py = spacing * y + simParams_.boundary_min_y + simParams_.sph_particle_radius;
            pz = simParams_.boundary_max_z + simParams_.sph_particle_radius;

            sph_boundary_pos[index] = MAKE_REAL4(px, py, pz, 0.0f);


            // Back plane
            px = spacing * x + simParams_.boundary_min_x - simParams_.sph_particle_radius;
            py = spacing * y + simParams_.boundary_min_y + simParams_.sph_particle_radius;
            pz = simParams_.boundary_min_z - simParams_.sph_particle_radius;

            sph_boundary_pos[index + num_plane_particles] = MAKE_REAL4(px, py, pz, 0.0f);

            index++;
        }
    }

    index += num_plane_particles;

    simParams_.boundaryPlaneLeftSideIndex = index;
    simParams_.boundaryPlaneLeftSideSize = num_plane_particles;
    simParams_.boundaryPlaneRightSideIndex = index + num_plane_particles;
    simParams_.boundaryPlaneRightSideSize = num_plane_particles;

    for (int y = 0; y < plane_size; y++)
    {
        for (int z = 0; z < plane_size + 1; z++)
        {
            // Left side plane
            px = simParams_.boundary_min_x - simParams_.sph_particle_radius;
            py = spacing * y + simParams_.boundary_min_y + simParams_.sph_particle_radius;
            pz = spacing * z + simParams_.boundary_min_z + simParams_.sph_particle_radius;

            sph_boundary_pos[index] = MAKE_REAL4(px, py, pz, 0.0f);

            // Right side plane
            px = simParams_.boundary_max_x + simParams_.sph_particle_radius;
            py = spacing * y + simParams_.boundary_min_y + simParams_.sph_particle_radius;
            pz = spacing * z + simParams_.boundary_min_z - simParams_.sph_particle_radius;

            sph_boundary_pos[index + num_plane_particles] = MAKE_REAL4(px, py, pz, 0.0f);

            index++;
        }
    }


    std::cout << Monitor::PADDING << "Boundary planes size: " << plane_size <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Bottom plane index: " <<
              simParams_.boundaryPlaneBottomIndex << ", size: " << simParams_.boundaryPlaneBottomSize <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Top plane index: " <<
              simParams_.boundaryPlaneTopIndex << ", size: " << simParams_.boundaryPlaneTopSize <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Front plane index: " <<
              simParams_.boundaryPlaneFrontIndex << ", size: " << simParams_.boundaryPlaneFrontSize <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Back plane index: " <<
              simParams_.boundaryPlaneBackIndex << ", size: " << simParams_.boundaryPlaneBackSize <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Left plane index: " <<
              simParams_.boundaryPlaneLeftSideIndex << ", size; " <<
              simParams_.boundaryPlaneLeftSideSize <<
              std::endl;
    std::cout << Monitor::PADDING << Monitor::PADDING << "Right plane index: " <<
              simParams_.boundaryPlaneRightSideIndex << ", size: " <<
              simParams_.boundaryPlaneRightSideSize <<
              std::endl;
    /////////////////////////////////////////////////////////////////
    // calculate rest density
    real_t dist_sq;
    double sumW = 0;

    int span = (int)ceil(simParams_.sph_kernel_coeff / 2);

    for (int z = -span; z <= span; ++z)
    {
        for (int y = -span; y <= span; ++y)
        {
            for (int x = -span; x <= span; ++x)
            {
                px = 2 * x * simParams_.sph_particle_radius;
                py = 2 * y * simParams_.sph_particle_radius;
                pz = 2 * z * simParams_.sph_particle_radius;

                dist_sq = px * px + py * py + pz * pz;

                sumW += kernel_poly6(dist_sq / simParams_.sph_kernel_smooth_length_squared);

            }

        }
    }

    simParams_.sph_rest_density = (real_t) sumW * simParams_.sph_particle_mass *
                                  simParams_.sph_kernel_poly6;

    std::cout << Monitor::PADDING << "SPH rest density: " << simParams_.sph_rest_density <<
              std::endl;
}

//------------------------------------------------------------------------------------------
void Scene::initPeridynamicsParticles(int* pd_activity,
                                      real4_t* pd_position,
                                      real4_t *pd_velocity)
{
    if(simParams_.num_pd_particle == 0)
    {
        return;
    }

    memcpy(pd_position, pd_position_cache_, sizeof(real4_t)*simParams_.num_pd_particle);


    real4_t translation = MAKE_REAL4(runningParams_.mesh_translation_x /
                                     simParams_.scaleFactor,
                                     runningParams_.mesh_translation_y / simParams_.scaleFactor,
                                     runningParams_.mesh_translation_z / simParams_.scaleFactor,
                                     0.0f);
    real4_t rotation = MAKE_REAL4(0.0, 0.0f, 0.0f, 0.0f);
//    real4_t rotation = MAKE_REAL4(0.0, M_PI / 6.0f, 0.0f, 0.0f);

    std::cout << Monitor::PADDING << "PD particles translated by: " <<
              translation.x << ", " << translation.y << ", " << translation.z << std::endl;
    std::cout << Monitor::PADDING << "PD particles rotated by: " <<
              rotation.x << " degree around axis: " <<
              rotation.y << ", " << rotation.z << ", " << rotation.w << std::endl;

    transformParticles(pd_position, translation, rotation, simParams_.num_pd_particle);

    // set the activity
    for(int i = 0; i < simParams_.num_pd_particle; ++i)
    {
        pd_activity[i] = ACTIVE;
        pd_velocity[i] = MAKE_REAL4_FROM_REAL(runningParams_.pd_initial_velocity);
    }

}

//------------------------------------------------------------------------------------------
void Scene::fillParticles(real4_t* particles, int* grid3d, real_t* margin3d,
                          real_t border,
                          real_t spacing, real_t jitter, int num_particles, bool position_correction)
{
    real_t pX, pY, pZ;


    for (int z = 0; z < grid3d[2]; z++)
    {
        for (int y = 0; y < grid3d[1]; y++)
        {
            for (int x = 0; x < grid3d[0]; x++)
            {
                int i = (z * grid3d[1] * grid3d[0]) + (y * grid3d[0]) + x;

                if (i >= num_particles)
                {
                    continue;
                }

                pX = simParams_.boundary_min_x + margin3d[0] + x * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;
                pY = simParams_.boundary_min_y + margin3d[1] + y * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;
                pZ = simParams_.boundary_min_z + margin3d[2] + z * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;

                // Correction of position
                if(position_correction)
                {
                    if(pX > simParams_.boundary_min_x)
                    {
                        if(pX < simParams_.boundary_min_x + border)
                        {
                            pX = simParams_.boundary_min_x + border;
                        }
                    }
                    else
                    {
                        if(pX > simParams_.boundary_min_x - border)
                        {
                            pX = simParams_.boundary_min_x - border;
                        }
                    }

                    if(pX < simParams_.boundary_max_x)
                    {
                        if(pX > simParams_.boundary_max_x - border)
                        {
                            pX = simParams_.boundary_max_x - border;
                        }
                    }
                    else
                    {
                        if(pX < simParams_.boundary_max_x + border)
                        {
                            pX = simParams_.boundary_max_x + border;
                        }
                    }

                    if(pY > simParams_.boundary_min_y)
                    {
                        if(pY < simParams_.boundary_min_y + border)
                        {
                            pY = simParams_.boundary_min_y + border;
                        }
                    }
                    else
                    {
                        if(pY > simParams_.boundary_min_y - border)
                        {
                            pY = simParams_.boundary_min_y - border;
                        }
                    }

                    if(pY < simParams_.boundary_max_y)
                    {
                        if(pY > simParams_.boundary_max_y - border)
                        {
                            pY = simParams_.boundary_max_y - border;
                        }
                    }
                    else
                    {
                        if(pY < simParams_.boundary_max_y + border)
                        {
                            pY = simParams_.boundary_max_y + border;
                        }
                    }

                    if(pZ > simParams_.boundary_min_z)
                    {
                        if(pZ < simParams_.boundary_min_z + border)
                        {
                            pZ = simParams_.boundary_min_z + border;
                        }
                    }
                    else
                    {
                        if(pZ > simParams_.boundary_min_z - border)
                        {
                            pZ = simParams_.boundary_min_z - border;
                        }
                    }



                    if(pZ < simParams_.boundary_max_z)
                    {
                        if(pZ > simParams_.boundary_max_z - border)
                        {
                            pZ = simParams_.boundary_max_z - border;
                        }
                    }
                    else
                    {
                        if(pZ < simParams_.boundary_max_z + border)
                        {
                            pZ = simParams_.boundary_max_z + border;
                        }
                    }

                }

                particles[i] = MAKE_REAL4(pX, pY, pZ, 0.0f);
//                printf("p: %d, %f, %f, %f\n", i, pX, pY, pZ);
            }
        }
    }
}

//------------------------------------------------------------------------------------------
int Scene::fillParticlesToMesh(MeshObject* meshObject, cyPoint3f box_min,
                               cyPoint3f box_max, real4_t* particles, int* grid3d, real_t spacing, real_t jitter,
                               int max_num_particles)
{
    real_t pX, pY, pZ;
    int num_particles = 0;
    double point[3];

//    bool y0=false;
    for (int z = 0; z < grid3d[2]; z++)
    {
        for (int y = 0; y < grid3d[1]; y++)
        {
            for (int x = 0; x < grid3d[0]; x++)
            {
                int i = (z * grid3d[1] * grid3d[0]) + (y * grid3d[0]) + x;

                if (i >= max_num_particles)
                {
                    continue;
                }

//                if(y0 && y==1)
//                    continue;
//                if(count == 3)
//                    continue;

                pX = box_min.x + x * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;
                pY = box_min.y + y * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;
                pZ = box_min.z + z * spacing
                     + (frand() * 2.0f - 1.0f) * jitter;

                point[0] = (double)pX;
                point[1] = (double)pY;
                point[2] = (double)pZ;

                if(point_inside_mesh(point, meshObject))
                {
                    // Correction of position
                    pX = (pX < box_min.x) ? box_min.x : pX;
                    pX = (pX > box_max.x) ? box_max.x : pX;

                    pY = (pY < box_min.y) ? box_min.y : pY;
                    pY = (pY > box_max.y) ? box_max.y : pY;

                    pZ = (pZ < box_min.z) ? box_min.z : pZ;
                    pZ = (pZ > box_max.z) ? box_max.z : pZ;
//if(!y0 && y==1) y0 = true;

//printf("P: %d, %d, %d, %f, %f, %f\n", x, y, z, pX, pY, pZ);

                    particles[num_particles] = MAKE_REAL4(pX, pY, pZ, 0.0);
                    ++num_particles;

                }
            }
        }
    }



    std::cout << Monitor::PADDING << "Total filled Peridynamics particles: " << num_particles
              <<
              std::endl;

    return num_particles;
}

//------------------------------------------------------------------------------------------
void Scene::fillTubeParticles(real4_t* particles, int tube_radius, real3_t base_center,
                              int* up_direction, real_t spacing, real_t jitter, int num_particles)
{

}

//------------------------------------------------------------------------------------------
void Scene::createSPHGrid(int* grid3d)
{
    grid3d[0] = (int) floor((simParams_.boundary_max_x - simParams_.boundary_min_x) /
                            (2.0f * simParams_.sph_particle_radius)) - 10;
    grid3d[1] = (int) floor((simParams_.boundary_max_y - simParams_.boundary_min_y) /
                            (2.0f * simParams_.sph_particle_radius)) - 3;
    grid3d[2] = (int) floor((simParams_.boundary_max_z - simParams_.boundary_min_z) /
                            (2.0f * simParams_.sph_particle_radius));

    std::cout << Monitor::PADDING << "Maximum SPH grid: " << grid3d[0] << "x" << grid3d[1] <<
              "x" << grid3d[2] << std::endl;
}

//------------------------------------------------------------------------------------------
void Scene::createPeridynamicsGrid(int* grid3d)
{
    grid3d[0] = (int) floor((simParams_.boundary_max_x - simParams_.boundary_min_x) /
                            (2.0f * simParams_.pd_particle_radius));
    grid3d[1] = (int) floor((simParams_.boundary_max_y - simParams_.boundary_min_y) /
                            (2.0f * simParams_.pd_particle_radius));
    grid3d[2] = (int) floor((simParams_.boundary_max_z - simParams_.boundary_min_z) /
                            (2.0f * simParams_.pd_particle_radius));
    std::cout << Monitor::PADDING << "Maximum Peridynamics grid: " << grid3d[0] << "x" <<
              grid3d[1] << "x" << grid3d[2] << std::endl;
}

//------------------------------------------------------------------------------------------
void Scene::createPeridynamicsGrid(cyPoint3f box_min, cyPoint3f box_max, int* grid3d)
{
    grid3d[0] = (int) ceil((box_max.x - box_min.x) /
                           (2.0f * simParams_.pd_particle_radius)) + 1;
    grid3d[1] = (int) ceil((box_max.y - box_min.y) /
                           (2.0f * simParams_.pd_particle_radius)) + 1;
    grid3d[2] = (int) ceil((box_max.z - box_min.z) /
                           (2.0f * simParams_.pd_particle_radius)) + 1;

    std::cout << Monitor::PADDING << "Maximum Peridynamics grid: " << grid3d[0] << "x" <<
              grid3d[1] << "x" << grid3d[2] << std::endl;
}

//------------------------------------------------------------------------------------------
inline double dot3(double a[3], real4_t b)
{
    return (a[0] * b.x + a[1] * b.y + a[2] * b.z);
}
void Scene::transformParticles(real4_t* particles, real4_t translation, real4_t rotation,
                               int num_particles)
{
    int i, j;
    double azimuth = rotation.x;
    double elevation = rotation.y;
    double roll = rotation.z;
    double sinA, cosA, sinE, cosE, sinR, cosR;
    double R[3][3];
    double tmp[4];
    sinA = std::sin(azimuth);
    cosA = std::cos(azimuth);
    sinE = std::sin(elevation);
    cosE = std::cos(elevation);
    sinR = std::sin(roll);
    cosR = std::cos(roll);

    R[0][0] = cosR * cosA - sinR * sinA * sinE;
    R[0][1] = sinR * cosA + cosR * sinA * sinE;
    R[0][2] = -sinA * cosE;

    R[1][0] = -sinR * cosE;
    R[1][1] = cosR * cosE;
    R[1][2] = sinE;

    R[2][0] = cosR * sinA + sinR * cosA * sinE;
    R[2][1] = sinR * sinA - cosR * cosA * sinE;
    R[2][2] = cosA * cosE;

    for (i = 0; i < num_particles; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            tmp[j] = dot3(R[j], particles[i]);
        }

        particles[i] = MAKE_REAL4(tmp[0] + translation.x, tmp[1] + translation.y,
                                  tmp[2] + translation.z, 0.0f);
    }
}

