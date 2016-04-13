//------------------------------------------------------------------------------------------
//
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QtWidgets>
#include "renderer.h"
#include "datareader.h"
#include "colorselector.h"
class Controller: public QWidget
{
    Q_OBJECT
    friend class MainWindow;
public:
    explicit Controller(QWidget* _parent);
    ~Controller();

    void setSimulationInfo(QList<QString>* _simInfo);
    void resetParameters();

public slots:
    void prevEnvTexture();
    void nextEnvTexture();
    void prevFloorTexture();
    void nextFloorTexture();

private:
    void setupGUI();

    QSignalMapper* signalMapperParticleViewing;
    QSignalMapper* signalMapperParticleColor;

    QSlider* sldFrameTime;
    QSlider* sldFrameStride;

    QComboBox* cbEnvTexture;
    QComboBox* cbFloorTexture;
    ColorSelector* wgSPHParticleColor;
    ColorSelector* wgPDParticleColor;

    QPushButton* btnPause;
    QPushButton* btnNextFrame;
    QPushButton* btnReset;
    QPushButton* btnReverse;
    QPushButton* btnRepeatPlay;
    QPushButton* btnClipYZPlane;
    QPushButton* btnDualViewport;
    QPushButton* btnHideInvisibleParticles;

    QListWidget* listSimInfo;
    QLineEdit* txtDataFolder;
    QPushButton* btnBrowseDataFolder;
};

#endif // CONTROLLER_H
