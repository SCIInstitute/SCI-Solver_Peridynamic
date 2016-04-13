//------------------------------------------------------------------------------------------
//
//
// Created on: 3/8/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QtGui>
#include <QtWidgets>

#include <fstream>
#include <sstream>
#include "parameters.h"

ParameterLoader::ParameterLoader():
    listSimInfo(NULL),
    num_frames(0),
    frame_duration(10),
    frame_stride(1),
    data_dir("")
{
    dataDirWatcher = new QFileSystemWatcher(this);
    connect(dataDirWatcher, &QFileSystemWatcher::directoryChanged,
            this, &ParameterLoader::countFiles);
}

//------------------------------------------------------------------------------------------
ParameterLoader::~ParameterLoader()
{
    delete listSimInfo;

}

//------------------------------------------------------------------------------------------
bool ParameterLoader::loadParams(QString data_path)
{
    if(!loadParameters(data_path.toStdString().c_str()))
    {
        return false;
    }

    if(data_dir != "")
    {
        dataDirWatcher->removePath(data_dir);
    }

    data_dir = data_path;
    dataDirWatcher->addPath(data_dir + "/SPH_POSITION");
    dataDirWatcher->addPath(data_dir + "/PD_POSITION");

    countFiles();
//    qDebug() << numFrames;

    return true;
}

//------------------------------------------------------------------------------------------
SimulationParameters ParameterLoader::getSimulationParameters()
{
    return simParams;
}

//------------------------------------------------------------------------------------------
bool ParameterLoader::loadParameters(const char* _dataPath)
{
    char fileName[128];
    sprintf(fileName, "%s/viz_info.dat", _dataPath);
    std::ifstream inFile(fileName);

    if (!inFile.is_open())
    {
        QMessageBox::critical(NULL, "Error", "Cannot read viz info from the data path.");
        return false;
    }


    std::string line;
    std::string paramName, paramValue;

    while (std::getline(inFile, line))
    {
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        if(line == "")
        {
            continue;
        }

        if(line.find("//") != std::string::npos)
        {
            continue;
        }

        std::istringstream iss(line);

        iss >> paramName >> paramValue;


        if(paramName == "num_total_particle")
        {
            simParams.num_total_particle = atoi(paramValue.c_str());
        }

        if(paramName == "num_sph_particle")
        {
            simParams.num_sph_particle = atoi(paramValue.c_str());
        }

        if(paramName == "num_pd_particle")
        {
            simParams.num_pd_particle = atoi(paramValue.c_str());
        }


        if(paramName == "sph_kernel_coeff")
        {
            simParams.sph_kernel_coeff = atof(paramValue.c_str());
        }

        if(paramName == "pd_kernel_coeff")
        {
            simParams.pd_kernel_coeff = atof(paramValue.c_str());
        }


        if(paramName == "boundary_min_x")
        {
            simParams.boundary_min_x = atof(paramValue.c_str());
        }

        if(paramName == "boundary_min_y")
        {
            simParams.boundary_min_y = atof(paramValue.c_str());
        }


        if(paramName == "boundary_min_z")
        {
            simParams.boundary_min_z = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_x")
        {
            simParams.boundary_max_x = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_y")
        {
            simParams.boundary_max_y = atof(paramValue.c_str());
        }

        if(paramName == "boundary_max_z")
        {
            simParams.boundary_max_z = atof(paramValue.c_str());
        }


        if(paramName == "sph_particle_mass")
        {
            simParams.sph_particle_mass = atof(paramValue.c_str());
        }

        if(paramName == "pd_particle_mass")
        {
            simParams.pd_particle_mass = atof(paramValue.c_str());
        }

        if(paramName == "sph_sph_viscosity")
        {
            simParams.sph_sph_viscosity = atof(paramValue.c_str());
        }


        if(paramName == "sph_boundary_viscosity")
        {
            simParams.sph_boundary_viscosity = atof(paramValue.c_str());
        }

        if(paramName == "sph_pd_slip")
        {
            simParams.sph_pd_slip = atof(paramValue.c_str());
        }

        if(paramName == "sph_particle_radius")
        {
            simParams.sph_particle_radius = atof(paramValue.c_str());
        }

        if(paramName == "sph_rest_density")
        {
            simParams.sph_rest_density = atof(paramValue.c_str());
        }

        if(paramName == "pd_particle_radius")
        {
            simParams.pd_particle_radius = atof(paramValue.c_str());
        }

        if(paramName == "pd_horizon")
        {
            simParams.pd_horizon = atof(paramValue.c_str());
        }

        if(paramName == "scaleFactor")
        {
            simParams.scaleFactor = atof(paramValue.c_str());
        }
    }

    inFile.close();

//    fread(simParams, 1, sizeof(SimulationParameter), fptr);
//    fclose(fptr);

    generateStringSimInfo();

    return true;
}

//------------------------------------------------------------------------------------------
void ParameterLoader::generateStringSimInfo()
{
    if(!listSimInfo)
    {
        listSimInfo = new QList<QString>;
    }

    listSimInfo->clear();

    listSimInfo->append(QString("Num. total particles: %1").arg(
                            simParams.num_total_particle));
    listSimInfo->append(QString("Num. fluid particles: %1").arg(simParams.num_sph_particle));
    listSimInfo->append(QString("Num. solid particles: %1").arg(simParams.num_pd_particle));
    listSimInfo->append(QString("Box min: %1, %2, %3").
                        arg(simParams.boundary_min_x * simParams.scaleFactor).
                        arg(simParams.boundary_min_y * simParams.scaleFactor).
                        arg(simParams.boundary_min_z * simParams.scaleFactor));
    listSimInfo->append(QString("Box max: %1, %2, %3").
                        arg(simParams.boundary_max_x * simParams.scaleFactor).
                        arg(simParams.boundary_max_y * simParams.scaleFactor).
                        arg(simParams.boundary_max_z * simParams.scaleFactor));
    listSimInfo->append(QString("SPH_kernel/SPH_radius: %1").
                        arg(simParams.sph_kernel_coeff));
    listSimInfo->append(QString("PD_kernel/PD_radius: %1").arg(simParams.pd_kernel_coeff));
    listSimInfo->append(QString("SPH mass: %1").arg(simParams.sph_particle_mass));
    listSimInfo->append(QString("PD mass: %1").arg(simParams.pd_particle_mass));
    listSimInfo->append(QString("SPH-SPH viscosity: %1").arg(simParams.sph_sph_viscosity));
    listSimInfo->append(QString("SPH-PD slip condition: %1").arg(simParams.sph_pd_slip));
    listSimInfo->append(QString("SPH-Boundary viscosity: %1").arg(
                            simParams.sph_boundary_viscosity));

}


//------------------------------------------------------------------------------------------
QList<QString>* ParameterLoader::getParametersAsString()
{
    return listSimInfo;
}

//------------------------------------------------------------------------------------------
void ParameterLoader::countFiles()
{
    QDir dataDir(data_dir);

    dataDir.cd("./SPH_POSITION");

    if(dataDir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).count() == 0)
    {
        dataDir.cd("../PD_POSITION");
    }


    num_frames = dataDir.entryList(QDir::NoDotAndDotDot | QDir::AllEntries).count();

qDebug() << num_frames;
    emit numFramesChanged();
}

