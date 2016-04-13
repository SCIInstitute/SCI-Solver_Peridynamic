//------------------------------------------------------------------------------------------
//
//
// Created on: 6/26/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef DUALVIEWPORTRENDERER_H
#define DUALVIEWPORTRENDERER_H

#include <QWidget>
#include "renderer.h"

#define NUM_VIEWPORTS 2
//------------------------------------------------------------------------------------------
class DualViewportRenderer : public QWidget
{
    Q_OBJECT
public:
    explicit DualViewportRenderer(QWidget *parent = 0);
    void allocateMemory(int num_sph_particles, int num_pd_particles,
                        float sph_particle_radius, float pd_particle_radius, int viewport);

    void enableDualViewport(bool status);

    void keyPressEvent(QKeyEvent*event);
    void keyReleaseEvent(QKeyEvent*event);


signals:

public slots:
    void update();
    void setSPHParticleColor(float r, float g, float b);
    void setPDParticleColor(float r, float g, float b);
    void enableImageOutput(bool status);
    void pauseImageOutput(bool status);
    void enableClipYZPlane(bool status);
    void hideInvisibleParticles(bool status);

    void setImageOutputPath(QString output_path);
    void resetCameraPosition();
    void setEnvironmentTexture(int texture);
    void setFloorTexture(int texture);
    void setParticleViewMode(int view_mode);
    void setParticleColorMode(int color_mode);

    void setParticlePositions(real_t* pd_positions, real_t* sph_positions,
                              int* pd_activities, int* sph_activities,
                              int current_frame,
                              int viewport);
    void setParticleDensitiesPositions(real_t* pd_positions, real_t* sph_positions,
                                       int* pd_activities, int* sph_activities,
                                       real_t* sph_densities, int current_frame, int viewport);
    void setParticleStiffnesPositions(real_t* pd_positions, real_t* sph_positions,
                                      int* pd_activities, int* sph_activities,
                                      real_t* pd_stiffness, int current_frame, int viewport);
    void setParticleActivitiesPositions(real_t* pd_positions, real_t* sph_positions,
                                        int* pd_activities, int* sph_activities, int current_frame, int viewport);


private:

    Renderer* renderers[2];

    bool bDualViewport;
};

#endif // DUALVIEWPORTRENDERER_H
