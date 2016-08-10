//------------------------------------------------------------------------------------------
//
//
// Created on: 6/26/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QtWidgets>
#include "dualviewport_renderer.h"

DualViewportRenderer::DualViewportRenderer(QWidget* parent) : QWidget(parent),
    bDualViewport(false)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport] = new Renderer(this, viewport);
    }

    connect(renderers[0], &Renderer::cameraChanged, renderers[1],
            &Renderer::setCamera);
    connect(renderers[1], &Renderer::cameraChanged, renderers[0],
            &Renderer::setCamera);


    renderers[1]->setVisible(false);

    QHBoxLayout* rendererLayout = new QHBoxLayout;
    rendererLayout->addWidget(renderers[0]);
    rendererLayout->addWidget(renderers[1]);

    setLayout(rendererLayout);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::allocateMemory(int num_sph_particles, int num_pd_particles,
                                          float sph_particle_radius, float pd_particle_radius,
                                          int viewport)
{
    renderers[viewport]->allocateMemory(num_sph_particles, num_pd_particles,
                                        sph_particle_radius, pd_particle_radius);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::enableDualViewport(bool status)
{
    bDualViewport = status;

    renderers[1]->setVisible(status);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::keyPressEvent(QKeyEvent* event)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->keyPressEvent(event);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::keyReleaseEvent(QKeyEvent* event)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->keyReleaseEvent(event);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::update()
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->update();
    }

//    this->update();
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setSPHParticleColor(float r, float g, float b)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setSPHParticleColor(r, g, b);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setPDParticleColor(float r, float g, float b)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setPDParticleColor(r, g, b);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::enableImageOutput(bool status)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->enableImageOutput(status);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::pauseImageOutput(bool status)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->pauseImageOutput(status);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::enableClipYZPlane(bool status)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->enableClipYZPlane(status);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::hideInvisibleParticles(bool status)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->hideInvisibleParticles(status);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setImageOutputPath(QString output_path)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setImageOutputPath(output_path);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::resetCameraPosition()
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->resetCameraPosition();
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setEnvironmentTexture(int texture)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setEnvironmentTexture(texture);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setFloorTexture(int texture)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setFloorTexture(texture);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticleViewMode(int view_mode)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setParticleViewMode(view_mode);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticleColorMode(int color_mode)
{
    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        renderers[viewport]->setParticleColorMode(color_mode);
    }
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticlePositions(real_t* pd_positions,
                                                real_t* sph_positions,
                                                int* pd_activities, int* sph_activities,
                                                int current_frame,
                                                int viewport)
{
    renderers[viewport]->setParticlePositions(pd_positions, sph_positions, pd_activities,
                                              sph_activities, current_frame);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticleDensitiesPositions(real_t* pd_positions,
                                                         real_t* sph_positions, int* pd_activities,
                                                         int* sph_activities,
                                                         real_t* sph_densities,
                                                         int current_frame, int viewport)
{
    renderers[viewport]->setParticleDensitiesPositions(pd_positions, sph_positions,
                                                       pd_activities, sph_activities,
                                                       sph_densities, current_frame);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticleStiffnesPositions(real_t* pd_positions,
                                                        real_t* sph_positions, int* pd_activities, int* sph_activities,
                                                        real_t* pd_stiffness,
                                                        int current_frame, int viewport)
{
    renderers[viewport]->setParticleStiffnesPositions(pd_positions, sph_positions,
                                                      pd_activities, sph_activities,
                                                      pd_stiffness, current_frame);
}

//------------------------------------------------------------------------------------------
void DualViewportRenderer::setParticleActivitiesPositions(real_t* pd_positions,
                                                          real_t* sph_positions, int* pd_activities,
                                                          int* sph_activities, int current_frame,
                                                          int viewport)
{
    renderers[viewport]->setParticleActivitiesPositions(pd_positions, sph_positions,
                                                        pd_activities, sph_activities, current_frame);
}
