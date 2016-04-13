//------------------------------------------------------------------------------------------
// mainwindow.h
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QtWidgets>
#include <QEvent>
//#include <QGLPixelBuffer>

#include "dualviewport_renderer.h"
#include "controller.h"
#include "parameters.h"
#include "datareader.h"

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1200

class MainWindow: public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = 0);
    ~MainWindow();

protected:
    void keyPressEvent(QKeyEvent*);
    void keyReleaseEvent(QKeyEvent*);

signals:
    void outputPathChanged(QString output_dir);

public slots:
    void browseDataPathVP0();
    void browseDataPathVP1();
    bool browseOutputPath();
    void enableImageOutput(bool status);
    void updateStatusCurrentFrame(int current_frame);
    void updateStatusNumFrames(int num_frames_);
    void updateNumFrames();
    void enableDualViewport(bool status);

private:
    enum StatusType
    {
        STATUS_CURRENT_FRAME,
        STATUS_NUM_FRAMES,
        STATUS_NUM_PARTICLES
    };

    void setupGUI();
    void setStatus(StatusType status_type, QString str);
    void startVisualizeData(int viewport);
    void updateStatusNumParticles(int num_sph_particles, int num_pd_particles, int viewport);

    ParameterLoader* parameterLoader[2];
    DataReader* dataReader;
    DualViewportRenderer* renderer;
    QSlider* sldProgress;
    QLineEdit* txtDataPath[2];
    QLineEdit* txtOutputPath;
    QToolButton* btnBrowseVP1;

    Controller* controller;

    QLabel* lblStatusNumParticles;
    QLabel* lblStatusCurrentFrame;
    QLabel* lblStatusNumFrames;

    SimulationParameters simParams_[2];
    int num_frames_;
    int num_viewports_;
    QString data_dirs_[2];

    friend class Controller;
};

#endif // MAINWINDOW_H
