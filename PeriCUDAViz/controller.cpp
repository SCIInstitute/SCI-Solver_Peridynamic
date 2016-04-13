//------------------------------------------------------------------------------------------
//
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#include "controller.h"

Controller::Controller(QWidget* _parent):
    QWidget(_parent),
    signalMapperParticleColor(NULL),
    signalMapperParticleViewing(NULL)
{
    setupGUI();
}

//------------------------------------------------------------------------------------------
Controller::~Controller()
{

}

//------------------------------------------------------------------------------------------
void Controller::setSimulationInfo(QList<QString>* _simInfo)
{
    listSimInfo->clear();

    for(int i = 0; i < _simInfo->count(); ++i)
    {
        listSimInfo->addItem(_simInfo->at(i));
    }
}

//------------------------------------------------------------------------------------------
void Controller::resetParameters()
{
    sldFrameTime->setValue(1);
    sldFrameStride->setValue(1);
}

//------------------------------------------------------------------------------------------
void Controller::prevEnvTexture()
{
    int index = cbEnvTexture->currentIndex();

    if(index > 0)
    {
        cbEnvTexture->setCurrentIndex(index - 1);
    }
    else
    {
        cbEnvTexture->setCurrentIndex(cbEnvTexture->count() - 1);
    }
}

//------------------------------------------------------------------------------------------
void Controller::nextEnvTexture()
{
    int index = cbEnvTexture->currentIndex();

    if(index < cbEnvTexture->count() - 1)
    {
        cbEnvTexture->setCurrentIndex(index + 1);
    }
    else
    {
        cbEnvTexture->setCurrentIndex(0);
    }
}

//------------------------------------------------------------------------------------------
void Controller::prevFloorTexture()
{
    int index = cbFloorTexture->currentIndex();

    if(index > 0)
    {
        cbFloorTexture->setCurrentIndex(index - 1);
    }
    else
    {
        cbFloorTexture->setCurrentIndex(cbFloorTexture->count() - 1);
    }
}

//------------------------------------------------------------------------------------------
void Controller::nextFloorTexture()
{
    int index = cbFloorTexture->currentIndex();

    if(index < cbFloorTexture->count() - 1)
    {
        cbFloorTexture->setCurrentIndex(index + 1);
    }
    else
    {
        cbFloorTexture->setCurrentIndex(0);
    }
}

//------------------------------------------------------------------------------------------
void Controller::setupGUI()
{
    ////////////////////////////////////////////////////////////////////////////////
    // environment textures
    QGridLayout* envTextureLayout = new QGridLayout;

    cbEnvTexture = new QComboBox;
    envTextureLayout->addWidget(cbEnvTexture, 0, 0, 1, 3);
    cbEnvTexture->addItem("None");

    cbEnvTexture->addItem("Sky1");
//    cbEnvTexture->addItem("Sky2");
//    cbEnvTexture->addItem("Sky3");

    QToolButton* btnPreviousEnvTexture = new QToolButton;
    btnPreviousEnvTexture->setArrowType(Qt::LeftArrow);
    envTextureLayout->addWidget(btnPreviousEnvTexture, 0, 3, 1, 1);

    QToolButton* btnNextEnvTexture = new QToolButton;
    btnNextEnvTexture->setArrowType(Qt::RightArrow);
    envTextureLayout->addWidget(btnNextEnvTexture, 0, 4, 1, 1);

    connect(btnPreviousEnvTexture, SIGNAL(clicked()), this, SLOT(prevEnvTexture()));
    connect(btnNextEnvTexture, SIGNAL(clicked()), this, SLOT(nextEnvTexture()));

    QGroupBox* envTextureGroup = new QGroupBox("Background");
    envTextureGroup->setLayout(envTextureLayout);

    ////////////////////////////////////////////////////////////////////////////////
    // floor textures
    QGridLayout* floorTextureLayout = new QGridLayout;

    cbFloorTexture = new QComboBox;
    floorTextureLayout->addWidget(cbFloorTexture, 0, 0, 1, 3);
    cbFloorTexture->addItem("None");
    cbFloorTexture->addItem("Checkerboard 1");

    QToolButton* btnPreviousFloorTexture = new QToolButton;
    btnPreviousFloorTexture->setArrowType(Qt::LeftArrow);
    floorTextureLayout->addWidget(btnPreviousFloorTexture, 0, 3, 1, 1);

    QToolButton* btnNextFloorTexture = new QToolButton;
    btnNextFloorTexture->setArrowType(Qt::RightArrow);
    floorTextureLayout->addWidget(btnNextFloorTexture, 0, 4, 1, 1);

    connect(btnPreviousFloorTexture, SIGNAL(clicked()), this, SLOT(prevFloorTexture()));
    connect(btnNextFloorTexture, SIGNAL(clicked()), this, SLOT(nextFloorTexture()));

//    cbFloorTexture->setCurrentIndex(1);
    QGroupBox* floorTextureGroup = new QGroupBox("Floor");
    floorTextureGroup->setLayout(floorTextureLayout);

    ////////////////////////////////////////////////////////////////////////////////
    // frame time
    sldFrameTime = new QSlider(Qt::Horizontal);
    sldFrameTime ->setRange(1, 100);
    sldFrameTime ->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    QSpinBox* spFrameTime = new QSpinBox;
    spFrameTime->setRange(1, 100);

    connect(sldFrameTime, &QSlider::valueChanged, spFrameTime, &QSpinBox::setValue);
    connect(spFrameTime, SIGNAL(valueChanged(int)), sldFrameTime, SLOT(setValue(int)));

    QGridLayout* frameTimeLayout = new QGridLayout;
    frameTimeLayout->addWidget(sldFrameTime, 0, 0,  1, 5);
    frameTimeLayout->addWidget(spFrameTime, 0, 5, 1, 1);

    QGroupBox* frameTimeGroup = new QGroupBox("Frame Sleep(ms)");
    frameTimeGroup->setLayout(frameTimeLayout);


    ///////////////////////////////////////////////////////////////////////////////
    // frame stride
    sldFrameStride = new QSlider(Qt::Horizontal);
    sldFrameStride ->setRange(1, 100);
    sldFrameStride ->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);

    QSpinBox* spFrameStride = new QSpinBox;
    spFrameStride->setRange(1, 100);

    connect(sldFrameStride, &QSlider::valueChanged, spFrameStride, &QSpinBox::setValue);
    connect(spFrameStride, SIGNAL(valueChanged(int)), sldFrameStride, SLOT(setValue(int)));

    QGridLayout* frameStrideLayout = new QGridLayout;
    frameStrideLayout->addWidget(sldFrameStride, 0, 0, 1, 5);
    frameStrideLayout->addWidget(spFrameStride, 0, 5, 1, 1);

    QGroupBox* frameStrideGroup = new QGroupBox("Frame Stride");
    frameStrideGroup->setLayout(frameStrideLayout);



    ////////////////////////////////////////////////////////////////////////////////
    /// color modes
    QRadioButton* rdbColorRandom = new QRadioButton(QString("Random"));
    QRadioButton* rdbColorRamp = new QRadioButton(QString("Ramp"));
    QRadioButton* rdbColorParticleType = new QRadioButton(QString("Particle Type"));
    QRadioButton* rdbColorDensity = new QRadioButton(QString("Density"));
    QRadioButton* rdbColorStiffness = new QRadioButton(QString("Stiffness"));
    QRadioButton* rdbColorActivity = new QRadioButton(QString("Activity"));


    rdbColorParticleType->setChecked(true);

    QGridLayout* colorModesLayout = new QGridLayout;
    colorModesLayout->addWidget(rdbColorRandom, 0, 0);
    colorModesLayout->addWidget(rdbColorRamp, 0, 1);
    colorModesLayout->addWidget(rdbColorParticleType, 1, 0);
    colorModesLayout->addWidget(rdbColorDensity, 1, 1);
    colorModesLayout->addWidget(rdbColorStiffness, 2, 0);
    colorModesLayout->addWidget(rdbColorActivity, 2, 1);


//    colorModesLayout->addWidget(rdbColorVelocity, 1, 1);


    QGroupBox* colorModesGroup = new QGroupBox;
    colorModesGroup->setTitle(tr("Color Mode"));
    colorModesGroup->setLayout(colorModesLayout);

    signalMapperParticleColor = new QSignalMapper(this);
    connect(rdbColorRandom, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));
    connect(rdbColorRamp, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));
    connect(rdbColorParticleType, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));
    connect(rdbColorDensity, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));
    connect(rdbColorStiffness, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));
    connect(rdbColorActivity, SIGNAL(clicked()), signalMapperParticleColor,
            SLOT(map()));


    signalMapperParticleColor->setMapping(rdbColorRandom,
                                          (int) ParticleColorMode::COLOR_RANDOM);
    signalMapperParticleColor->setMapping(rdbColorRamp, (int) ParticleColorMode::COLOR_RAMP);
    signalMapperParticleColor->setMapping(rdbColorParticleType,
                                          (int) ParticleColorMode::COLOR_PARTICLE_TYPE);
    signalMapperParticleColor->setMapping(rdbColorDensity, (int) ParticleColorMode::COLOR_DENSITY);
    signalMapperParticleColor->setMapping(rdbColorStiffness, (int) ParticleColorMode::COLOR_STIFFNESS);
    signalMapperParticleColor->setMapping(rdbColorActivity, (int) ParticleColorMode::COLOR_ACTIVITY);

    wgSPHParticleColor = new ColorSelector;
    wgSPHParticleColor->setAutoFillBackground(true);
    wgSPHParticleColor->setColor(QColor(209, 115, 255));

    wgPDParticleColor = new ColorSelector;
    wgPDParticleColor->setAutoFillBackground(true);
    wgPDParticleColor->setColor(QColor(107, 255, 128));

    QGridLayout* particleColorLayout = new QGridLayout;
    particleColorLayout->addWidget(new QLabel("SPH Particle:"), 0, 0, Qt::AlignRight);
    particleColorLayout->addWidget(wgSPHParticleColor, 0, 1, 1, 2);
    particleColorLayout->addWidget(new QLabel("PD Particle:"), 1, 0, Qt::AlignRight);
    particleColorLayout->addWidget(wgPDParticleColor, 1, 1, 1, 2);

    QGroupBox* particleColorGroup = new QGroupBox("Particles' Color");
    particleColorGroup->setLayout(particleColorLayout);

    ////////////////////////////////////////////////////////////////////////////////
    /// particle viewing modes
    ////////////////////////////////////////////////////////////////////////////////
    QRadioButton* rdbPVSphere = new QRadioButton(QString("Sphere"));
    QRadioButton* rdbPVPoint = new QRadioButton(QString("Point"));
    QRadioButton* rdbPVOpaqueSurface = new QRadioButton(QString("Surface Only"));
    QRadioButton* rdbPVTransparentSurface = new QRadioButton(QString("Surface & Texture"));

    rdbPVSphere->setChecked(true);

    QGridLayout* pViewingModesLayout = new QGridLayout;
    pViewingModesLayout->addWidget(rdbPVSphere, 0, 0);
    pViewingModesLayout->addWidget(rdbPVPoint, 1, 0);

    pViewingModesLayout->addWidget(rdbPVOpaqueSurface, 0, 1);
    pViewingModesLayout->addWidget(rdbPVTransparentSurface, 1, 1);


    QGroupBox* pViewingModesGroup = new QGroupBox;
    pViewingModesGroup->setTitle(tr("Particle Viewing Mode"));
    pViewingModesGroup->setLayout(pViewingModesLayout);


    signalMapperParticleViewing = new QSignalMapper(this);
    connect(rdbPVSphere, SIGNAL(clicked(bool)), signalMapperParticleViewing,
            SLOT(map()));
    connect(rdbPVPoint, SIGNAL(clicked(bool)), signalMapperParticleViewing,
            SLOT(map()));
    connect(rdbPVOpaqueSurface, SIGNAL(clicked(bool)), signalMapperParticleViewing,
            SLOT(map()));
    connect(rdbPVTransparentSurface, SIGNAL(clicked(bool)), signalMapperParticleViewing,
            SLOT(map()));

    signalMapperParticleViewing->setMapping(rdbPVSphere,
                                            (int) ParticleViewMode::SPHERES_VIEW);
    signalMapperParticleViewing->setMapping(rdbPVPoint, (int) ParticleViewMode::POINTS_VIEW);
    signalMapperParticleViewing->setMapping(rdbPVOpaqueSurface,
                                            (int) ParticleViewMode::OPAQUE_SURFACE_VIEW);
    signalMapperParticleViewing->setMapping(rdbPVTransparentSurface,
                                            (int) ParticleViewMode::TRANSPARENT_SURFACE_VIEW);

    ////////////////////////////////////////////////////////////////////////////////
    /// simulation info
    ////////////////////////////////////////////////////////////////////////////////
    QVBoxLayout* simInfoLayout = new QVBoxLayout;
    simInfoLayout->setContentsMargins(0, 0, 0, 0);


    listSimInfo = new QListWidget;

//    for(int i = 0; i < 100; ++i)
//    {
//        listSimInfo->addItem(QString("Item: %1").arg(i));
//    }

    simInfoLayout->addWidget(listSimInfo);

    QGroupBox* simInfoGroup = new QGroupBox;
    simInfoGroup->setTitle("Simulation info");
    simInfoGroup->setLayout(simInfoLayout);


    ////////////////////////////////////////////////////////////////////////////////
    /// buttons
    ////////////////////////////////////////////////////////////////////////////////

    btnPause = new QPushButton(QString("Pause"));
    btnPause->setCheckable(true);
    btnNextFrame = new QPushButton(QString("Next Frame"));
    btnReset = new QPushButton(QString("Reset"));
    btnReverse = new QPushButton(QString("Reverse"));
    btnReverse->setCheckable(true);
    btnRepeatPlay = new QPushButton(QString("Repeat"));
    btnRepeatPlay->setCheckable(true);
    btnClipYZPlane= new QPushButton(QString("Clip YZ Plane"));
    btnClipYZPlane->setCheckable(true);
    btnDualViewport= new QPushButton(QString("Dual Viewport"));
    btnDualViewport->setCheckable(true);
    btnHideInvisibleParticles= new QPushButton(QString("Hide Invsb Particles"));
    btnHideInvisibleParticles->setCheckable(true);



    ////////////////////////////////////////////////////////////////////////////////
    /// controls's layout
    ////////////////////////////////////////////////////////////////////////////////

    QVBoxLayout* controlLayout = new QVBoxLayout;
    controlLayout->addWidget(envTextureGroup);
    controlLayout->addWidget(floorTextureGroup);
    controlLayout->addWidget(frameTimeGroup);
    controlLayout->addWidget(frameStrideGroup);
    controlLayout->addWidget(pViewingModesGroup);
    controlLayout->addWidget(colorModesGroup);
    controlLayout->addWidget(particleColorGroup);
    controlLayout->addWidget(simInfoGroup);
    controlLayout->addStretch();

    controlLayout->addWidget(btnPause);
//    controlLayout->addWidget(btnNextFrame);
    controlLayout->addWidget(btnReset);
    controlLayout->addWidget(btnRepeatPlay);
    controlLayout->addWidget(btnReverse);
    controlLayout->addWidget(btnClipYZPlane);
    controlLayout->addWidget(btnDualViewport);
    controlLayout->addWidget(btnHideInvisibleParticles);




    setLayout(controlLayout);
    setFixedWidth(300);
}

