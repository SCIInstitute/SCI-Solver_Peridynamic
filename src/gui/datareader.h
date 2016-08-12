//------------------------------------------------------------------------------------------
//
//
// Created on: 3/9/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef DATAREADER_H
#define DATAREADER_H

#include <QtWidgets>
#include "dualviewport_renderer.h"
#include "parameters.h"

class DataReader: public QObject
{
    Q_OBJECT
    friend class MainWindow;
public:

    DataReader();
    ~DataReader();

    void setSimulationDataInfo(QString* data_dir, SimulationParameters* simParams,
                               int num_frames, int num_viewports);

    void updateNumFrames(int num_frames_);
    void setCurrentFrame(int current_frame_);
    void resetToFirstFrame();


signals:
    void currentFrameChanged(int frame);
    void particlePositonsChanged(real_t* pd_particles, real_t* sph_particles,
                                 int* pd_activities, int* sph_activities,
                                 int current_frame, int viewport);
    void particleDensitiesPositonsChanged(real_t* pd_particles, real_t* sph_particles,
                                          int* pd_activities, int* sph_activities,
                                          real_t* particle_densities, int current_frame, int viewport);
    void particleStiffnesPositonsChanged(real_t* pd_particles, real_t* sph_particles,
                                         int* pd_activities, int* sph_activities,
                                         real_t* particle_stiffness, int current_frame, int viewport);
    void particleActivitiesPositonsChanged(real_t* pd_particles, real_t* sph_particles,
                                           int* pd_activities, int* sph_activities,
                                           int current_frame, int viewport);

public slots:
    void setFrameTime(int frame_time);
    void setFrameStride(int frame_stride);
    void enableRepeat(bool status);
    void enableReverse(bool status);
    void pause(bool status);
    void readNextFrameAutomatically();
    void readNextFrame();
    void setParticleColorMode(int color_mode);

    void hideInvisibleParticles(bool status);

private:

    void clearMemory();
    void allocateMemory();
    bool readFrame(int frame, int viewport);
    bool readData(const char* data_file, void* data, int data_size);
    bool readPosition(int frame, int viewport);
    bool readActivity(int frame, int viewport);
    bool readStiffness(int frame, int viewport);
    bool readDensity(int frame, int viewport);
    bool readSortedPosition(int frame, int viewport);

    int num_viewports_;
    SimulationParameters simParams_[2];
    QString data_dir_[2];
    std::string dir_;
    int num_frames_;
    int current_frame_;
    int frame_time_;
    int frame_stride_;
    ParticleColorMode currentColorMode_;

    real_t* sph_positions_[2];
    real_t* sph_densities_[2];
    real_t* pd_positions_[2];
    real_t* pd_stiffness_[2];
    int* sph_activities_[2];
    int* pd_activities_[2];

    QTimer* timer;
    bool bRepeat;
    bool bReverse;
    bool bPause;
    bool bHideInvisibleParticles;

};

#endif // DATAREADER_H
