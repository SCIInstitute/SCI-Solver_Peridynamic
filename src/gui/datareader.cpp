//------------------------------------------------------------------------------------------
//
//
// Created on: 3/9/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#include "datareader.h"

DataReader::DataReader():
    num_frames_(0),
    current_frame_(1),
    frame_time_(0),
    frame_stride_(1),
    bRepeat(false),
    bReverse(false),
    bPause(false),
    bHideInvisibleParticles(false),
    currentColorMode_(COLOR_PARTICLE_TYPE),
    num_viewports_(1)
{

    for(int viewport = 0; viewport < NUM_VIEWPORTS; ++viewport)
    {
        data_dir_[viewport] = QString("");
        sph_positions_[viewport] = NULL;
        sph_densities_[viewport] = NULL;
        sph_activities_[viewport] = NULL;
        pd_positions_[viewport] = NULL;
        pd_stiffness_[viewport] = NULL;
        pd_activities_[viewport] = NULL;
    }

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(readNextFrameAutomatically()));
    //timer->start(frameTime);
}

//------------------------------------------------------------------------------------------
DataReader::~DataReader()
{
    clearMemory();
}

//------------------------------------------------------------------------------------------
void DataReader::setSimulationDataInfo(QString* data_dir, SimulationParameters* simParams,
                                       int num_frames, int num_viewports)
{
    for(int i = 0; i < num_viewports_; ++i)
    {
        simParams_[i] = simParams[i];
        data_dir_[i] = data_dir[i];
    }
    this->dir_ = data_dir_[0].toStdString();

    num_frames_ = num_frames;
    num_viewports_ = num_viewports;

    // setup memory
    clearMemory();
    allocateMemory();

    resetToFirstFrame();
    timer->start(frame_time_);
}

//------------------------------------------------------------------------------------------
void DataReader::updateNumFrames(int num_frames)
{
    num_frames_ = num_frames;

    if(current_frame_ > num_frames)
    {
        setCurrentFrame(num_frames - 1);
    }
}

//------------------------------------------------------------------------------------------
void DataReader::setCurrentFrame(int current_frame)
{
    bool success = true;

    for(int viewport = 0; viewport < num_viewports_; ++viewport)
    {
        success = success && readFrame(current_frame, viewport);
    }

    if(success)
    {
        current_frame_ = current_frame;
        emit currentFrameChanged(current_frame);
    }
}

//------------------------------------------------------------------------------------------
void DataReader::readNextFrameAutomatically()
{
    if(bPause)
    {
        return;
    }

    readNextFrame();

}

//------------------------------------------------------------------------------------------
void DataReader::resetToFirstFrame()
{
    setCurrentFrame(1);
}

//------------------------------------------------------------------------------------------
void DataReader::setFrameTime(int frame_time)
{
    frame_time_ = frame_time;

    timer->setInterval(frame_time_);
}

//------------------------------------------------------------------------------------------
void DataReader::setFrameStride(int frame_stride)
{
    frame_stride_ = frame_stride;
}

//------------------------------------------------------------------------------------------
void DataReader::enableRepeat(bool status)
{
    bRepeat = status;
}

//------------------------------------------------------------------------------------------
void DataReader::enableReverse(bool status)
{
    bReverse = status;
}

//------------------------------------------------------------------------------------------
void DataReader::pause(bool status)
{
    bPause = status;
}

//------------------------------------------------------------------------------------------
void DataReader::readNextFrame()
{
    int nextFrame = current_frame_ + frame_stride_;

    if(bReverse)
    {
        nextFrame = current_frame_ - frame_stride_;
    }

    if(bRepeat)
    {
        if(nextFrame < 0)
        {
            nextFrame = num_frames_ - 1;
        }

        if(nextFrame >= num_frames_)
        {
            nextFrame = 1;
        }

        bool success = true;

        for(int viewport = 0; viewport < num_viewports_; ++viewport)
        {
            success = success && readFrame(nextFrame, viewport);
        }

        if(success)
        {
            current_frame_ = nextFrame;
            emit currentFrameChanged(current_frame_);
        }

    }

//    else if((nextFrame >= 0) && (nextFrame < numFrames))
    {
        bool success = true;

        for(int viewport = 0; viewport < num_viewports_; ++viewport)
        {
            success = success && readFrame(nextFrame, viewport);
        }

        if(success)
        {
            current_frame_ = nextFrame;
            emit currentFrameChanged(current_frame_);
        }
    }

}

//------------------------------------------------------------------------------------------
void DataReader::setParticleColorMode(int color_mode)
{
    currentColorMode_ = static_cast<ParticleColorMode>(color_mode);

    for(int viewport = 0; viewport < num_viewports_; ++viewport)
    {
        readFrame(current_frame_, viewport);
    }
}

//------------------------------------------------------------------------------------------
void DataReader::hideInvisibleParticles(bool status)
{
    bHideInvisibleParticles = status;
}

//------------------------------------------------------------------------------------------
//bool DataReader::isLastReadDone()
//{

//}

//------------------------------------------------------------------------------------------
void DataReader::clearMemory()
{
    for(int viewport = 0; viewport < num_viewports_; ++viewport)
    {
        if(pd_positions_[viewport])
        {
            delete[] pd_positions_[viewport];
            pd_positions_[viewport] = NULL;
        }

        if(sph_positions_[viewport])
        {
            delete[] sph_positions_[viewport];
            sph_positions_[viewport] = NULL;
        }

        if(sph_densities_[viewport])
        {
            delete[] sph_densities_[viewport];
            sph_densities_[viewport] = NULL;
        }

        if(pd_stiffness_[viewport])
        {
            delete[] pd_stiffness_[viewport];
            pd_stiffness_[viewport] = NULL;
        }

        if(pd_activities_[viewport])
        {
            delete[] pd_activities_[viewport];
            pd_activities_[viewport] = NULL;
        }

        if(sph_activities_[viewport])
        {
            delete[] sph_activities_[viewport];
            sph_activities_[viewport] = NULL;
        }
    }
}

//------------------------------------------------------------------------------------------
void DataReader::allocateMemory()
{
    for(int i = 0; i < num_viewports_; ++i)
    {
        sph_positions_[i] = new real_t[simParams_[i].num_sph_particle * 4];
        sph_densities_[i] = new real_t[simParams_[i].num_sph_particle];
        sph_activities_[i] = new int[simParams_[i].num_sph_particle];

        pd_positions_[i] = new real_t[simParams_[i].num_pd_particle * 4];
        pd_stiffness_[i] = new real_t[simParams_[i].num_pd_particle];
        pd_activities_[i] = new int[simParams_[i].num_pd_particle];
    }
}

//------------------------------------------------------------------------------------------
bool DataReader::readFrame(int frame, int viewport)
{

//    qDebug() << "current frame: " << _frame << "/" << numFrames;
//    qDebug() << currentParticleColorMode;
//        qDebug() << bHideInvisibleParticles;


    switch(currentColorMode_)
    {
    case COLOR_STIFFNESS:
    {

        if(readPosition(frame, viewport) && readStiffness(frame, viewport) &&
           (!bHideInvisibleParticles || (bHideInvisibleParticles && readActivity(frame, viewport))))
        {
            emit particleStiffnesPositonsChanged(pd_positions_[viewport],
                                                 sph_positions_[viewport],
                                                 pd_activities_[viewport], sph_activities_[viewport],
                                                 pd_stiffness_[viewport], frame, viewport);
            return true;
        }
        else
        {
            return false;
        }
    }
    break;

    case COLOR_ACTIVITY:
    {
        if(readPosition(frame, viewport) && readActivity(frame, viewport))
        {

            emit particleActivitiesPositonsChanged(pd_positions_[viewport],
                                                   sph_positions_[viewport],
                                                   pd_activities_[viewport], sph_activities_[viewport],
                                                   frame, viewport);
            return true;

        }
        else
        {
            return false;
        }
    }
    break;

    case COLOR_RANDOM:
    case COLOR_RAMP:
    case COLOR_PARTICLE_TYPE:
    {
        if(readPosition(frame, viewport) &&
           (!bHideInvisibleParticles || (bHideInvisibleParticles && readActivity(frame, viewport))))
        {
//            qDebug() << "read " << viewport << simParams_[viewport].num_pd_particle;
            emit particlePositonsChanged(pd_positions_[viewport], sph_positions_[viewport],
                                         pd_activities_[viewport], sph_activities_[viewport],
                                         frame, viewport);
            return true;
        }
        else
        {
            return false;
        }
    }
    break;

    case COLOR_DENSITY:
    {
        if(readSortedPosition(frame, viewport) && readDensity(frame, viewport) &&
           (!bHideInvisibleParticles || (bHideInvisibleParticles && readActivity(frame, viewport))))
        {
            emit particleDensitiesPositonsChanged(pd_positions_[viewport], sph_positions_[viewport],
                                                  pd_activities_[viewport], sph_activities_[viewport],
                                                  sph_densities_[viewport], frame, viewport);
            return true;
        }
        else
        {
            return false;
        }
    }
    break;

    default:
        return false;

    }



}

//------------------------------------------------------------------------------------------
bool DataReader::readData(const char* data_file, void* data, int data_size)
{
    if(data_size <= 0)
    {
        return true;
    }

    FILE* fptr;
    fptr = fopen(data_file, "r");

    if (!fptr)
    {
        return false;
    }

    fread(data, 1, data_size, fptr);
    fclose(fptr);

    return true;
}

//------------------------------------------------------------------------------------------
bool DataReader::readPosition(int frame, int viewport)
{
  auto s1 = this->dir_ + "/SPH_POSITION/frame." + std::to_string(frame);
  auto s2 = this->dir_ + "/PD_POSITION/frame." + std::to_string(frame);

    return( readData(s1.c_str(), sph_positions_[viewport],
                     4 * sizeof(real_t) * simParams_[viewport].num_sph_particle) &&
                     readData(s2.c_str(), pd_positions_[viewport],
                     4 * sizeof(real_t) * simParams_[viewport].num_pd_particle)  &&
            simParams_[viewport].num_total_particle > 0);


}

//------------------------------------------------------------------------------------------
bool DataReader::readActivity(int frame, int viewport)
{
    char fileNameSPH[128];
    char fileNamePD[128];

    sprintf(fileNameSPH, "%s/SPH_ACTIVITY/frame.%lu",
            data_dir_[viewport].toStdString().c_str(), frame);
    sprintf(fileNamePD, "%s/PD_ACTIVITY/frame.%lu", data_dir_[viewport].toStdString().c_str(),
            frame);

    return( readData(fileNameSPH, sph_activities_[viewport],
                     sizeof(int) * simParams_[viewport].num_sph_particle) &&
            readData(fileNamePD, pd_activities_[viewport],
                     sizeof(int) * simParams_[viewport].num_pd_particle)  &&
            simParams_[viewport].num_total_particle > 0);
}

//------------------------------------------------------------------------------------------
bool DataReader::readStiffness(int frame, int viewport)
{
    char fileNamePD[128];

    sprintf(fileNamePD, "%s/PD_BOND_STIFFNESS/frame.%lu",
            data_dir_[viewport].toStdString().c_str(),
            frame);

    return( readData(fileNamePD, pd_stiffness_[viewport],
                     sizeof(real_t) * simParams_[viewport].num_pd_particle)  &&
            simParams_[viewport].num_total_particle > 0);
}

//------------------------------------------------------------------------------------------
bool DataReader::readDensity(int frame, int viewport)
{
    char fileNameSPH[128];

    sprintf(fileNameSPH, "%s/SPH_SORTED_DENSITY/frame.%lu",
            data_dir_[viewport].toStdString().c_str(), frame);

    return( readData(fileNameSPH, sph_densities_[viewport],
                     sizeof(real_t) * simParams_[viewport].num_sph_particle) &&
            simParams_[viewport].num_total_particle > 0);
}

//------------------------------------------------------------------------------------------
bool DataReader::readSortedPosition(int frame, int viewport)
{
    char fileNameSPH[128];
    char fileNamePD[128];

    sprintf(fileNameSPH, "%s/SPH_SORTED_POSITION/frame.%lu",
            data_dir_[viewport].toStdString().c_str(),
            frame);
    sprintf(fileNamePD, "%s/PD_POSITION/frame.%lu", data_dir_[viewport].toStdString().c_str(),
            frame);

    return( readData(fileNameSPH, sph_positions_[viewport],
                     4 * sizeof(real_t) * simParams_[viewport].num_sph_particle) &&
            readData(fileNamePD, pd_positions_[viewport],
                     4 * sizeof(real_t) * simParams_[viewport].num_pd_particle)  &&
            simParams_[viewport].num_total_particle > 0);
}
