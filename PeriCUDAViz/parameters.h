//------------------------------------------------------------------------------------------
//
//
// Created on: 3/8/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef SIMULATIONPARAMETERS_H
#define SIMULATIONPARAMETERS_H

#include <QtWidgets>
#include <QFileSystemWatcher>
#include <QList>
#include <QString>
#include <stdint.h>

//#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
typedef float real_t;
#define GL_REAL GL_FLOAT
typedef GLfloat GLREAL;
#else // double precision
typedef double real_t;
#define GL_REAL GL_DOUBLE
typedef GLdouble GLREAL;
#endif


struct SimulationParameters
{
    // read from file
    int num_total_particle;
    int num_sph_particle;
    int num_pd_particle;

    real_t sph_kernel_coeff;
    real_t pd_kernel_coeff;
    real_t sph_particle_mass;
    real_t pd_particle_mass;
    real_t sph_sph_viscosity;
    real_t sph_boundary_viscosity;
    real_t sph_pd_slip;

    real_t boundary_min_x;
    real_t boundary_min_y;
    real_t boundary_min_z;
    real_t boundary_max_x;
    real_t boundary_max_y;
    real_t boundary_max_z;

    real_t sph_particle_radius;
    real_t sph_rest_density;
    real_t pd_particle_radius;
    real_t pd_horizon;

    real_t scaleFactor;

};


enum Activity
{
    ACTIVE = 0,
    SEMI_ACTIVE,
    INACTIVE,
    INVISIBLE,
    NUM_ACTIVITY_MODE
};


//------------------------------------------------------------------------------------------
class ParameterLoader: public QObject
{
    Q_OBJECT
public:
    ParameterLoader();
    ~ParameterLoader();

    bool loadParams(QString data_path);
    SimulationParameters getSimulationParameters();
    QList<QString>* getParametersAsString();

    int num_frames;
    int frame_duration;    // by milisecond
    int frame_stride;    // how many frames will be skip each render
public slots:
    void countFiles();

signals:
    void numFramesChanged();

private:
    bool loadParameters(const char* _dataDir);
    void generateStringSimInfo();

    SimulationParameters simParams;

    QString data_dir;
    QList<QString>* listSimInfo;
    QFileSystemWatcher* dataDirWatcher;
};

#endif // SIMULATIONPARAMETERS_H
