//------------------------------------------------------------------------------------------
// mainwindow.cpp
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include <QMouseEvent>
#include <qmath.h>
#include "controller.h"
#include "mainwindow.h"

MainWindow::MainWindow(QWidget* parent): QMainWindow(parent),
    num_frames_(0),
    num_viewports_(1)
{
    setupGUI();

    setWindowTitle("SPH-Peridynamics Simulation Visualization");
    setFocusPolicy(Qt::StrongFocus);

    parameterLoader[0] = new ParameterLoader;
    parameterLoader[1] = new ParameterLoader;
    dataReader = new DataReader;

    /////////////////////////////////////////////////////////////////
    // connect controller with renderer

    connect(controller->cbEnvTexture, SIGNAL(currentIndexChanged(int)), renderer,
            SLOT(setEnvironmentTexture(int)));
    connect(controller->cbFloorTexture, SIGNAL(currentIndexChanged(int)), renderer,
            SLOT(setFloorTexture(int)));

    connect(controller->sldFrameTime, &QSlider::valueChanged, dataReader,
            &DataReader::setFrameTime);
    connect(controller->sldFrameStride, &QSlider::valueChanged, dataReader,
            &DataReader::setFrameStride);

    connect(controller->signalMapperParticleViewing, SIGNAL(mapped(int)), renderer,
            SLOT(setParticleViewMode(int)));
    connect(controller->signalMapperParticleColor, SIGNAL(mapped(int)), renderer,
            SLOT(setParticleColorMode(int)));
    connect(controller->signalMapperParticleColor, SIGNAL(mapped(int)), dataReader,
            SLOT(setParticleColorMode(int)));
    connect(controller->wgSPHParticleColor, &ColorSelector::colorChanged, renderer,
            & DualViewportRenderer::setSPHParticleColor);
    connect(controller->wgPDParticleColor, &ColorSelector::colorChanged, renderer,
            & DualViewportRenderer::setPDParticleColor);

    connect(dataReader, SIGNAL(&DataReader::currentFrameChanged(int)),
      sldProgress, SLOT(&QSlider::setValue(int)));
    connect(dataReader, SIGNAL(&DataReader::currentFrameChanged(int)), this,
           SLOT( &MainWindow::updateStatusCurrentFrame(int)));

    connect(dataReader, SIGNAL(&DataReader::particlePositonsChanged(
      real_t*,real_t*,int*,int*,int,int)), renderer,
      SLOT(&DualViewportRenderer::setParticlePositions(
      real_t*, real_t*, int*, int*, int, int)));
    connect(dataReader, SIGNAL(&DataReader::particleDensitiesPositonsChanged(
      *, real_t*, int*, int*, real_t*, int, int)), renderer,
      SLOT(&DualViewportRenderer::setParticleDensitiesPositions(
      real_t*, real_t*, int*, int*, real_t*, int, int)));
    connect(dataReader, SIGNAL(&DataReader::particleStiffnesPositonsChanged(
      real_t*, real_t*, int*, int*, real_t*, int, int)), renderer,
      SLOT(&DualViewportRenderer::setParticleStiffnesPositions(
      real_t*, real_t*, int*, int*, real_t*, int, int)));
    connect(dataReader, SIGNAL(&DataReader::particleActivitiesPositonsChanged(
      real_t*, real_t*, int*, int*, int, int)), renderer,
      SLOT(&DualViewportRenderer::setParticleActivitiesPositions(
      real_t*, real_t*, int*, int*, int, int)));
    connect(sldProgress, SIGNAL(&QSlider::sliderMoved(int)), 
      dataReader, SLOT(&DataReader::setCurrentFrame(int)));


    /////////////////////////////////////////////////////////////////
    // start, stop
    connect(controller->btnPause, &QPushButton::clicked, dataReader, &DataReader::pause);
    connect(controller->btnPause, &QPushButton::clicked, renderer,
            & DualViewportRenderer::pauseImageOutput);
    connect(controller->btnNextFrame, &QPushButton::clicked, dataReader,
            &DataReader::readNextFrame);
    connect(controller->btnReset, &QPushButton::clicked, dataReader,
            &DataReader::resetToFirstFrame);
    connect(controller->btnRepeatPlay, &QPushButton::clicked, dataReader,
            &DataReader::enableRepeat);
    connect(controller->btnReverse, &QPushButton::clicked, dataReader,
            &DataReader::enableReverse);
    connect(controller->btnClipYZPlane, &QPushButton::clicked, renderer,
            & DualViewportRenderer::enableClipYZPlane);
    connect(controller->btnDualViewport, &QPushButton::clicked, renderer,
            & DualViewportRenderer::enableDualViewport);
    connect(controller->btnDualViewport, &QPushButton::clicked, this,
            &MainWindow::enableDualViewport);
    connect(controller->btnHideInvisibleParticles, &QPushButton::clicked, renderer,
            &DualViewportRenderer::hideInvisibleParticles);
    connect(controller->btnHideInvisibleParticles, &QPushButton::clicked, dataReader,
            &DataReader::hideInvisibleParticles);

    connect(parameterLoader[0], &ParameterLoader::numFramesChanged, this,
            &MainWindow::updateNumFrames);
    connect(parameterLoader[1], &ParameterLoader::numFramesChanged, this,
            &MainWindow::updateNumFrames);

    // Update continuously
    QTimer* timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), renderer, SLOT(update()));
    timer->start(0);

}

//------------------------------------------------------------------------------------------
MainWindow::~MainWindow()
{
    delete renderer;
    delete controller;
    delete lblStatusCurrentFrame;
}


//------------------------------------------------------------------------------------------
void MainWindow::setupGUI()
{
    renderer = new DualViewportRenderer(this);

    sldProgress = new QSlider(Qt::Horizontal);
    sldProgress->setRange(0, 1);
    sldProgress->setMinimum(0);
    sldProgress->setValue(0);
//    sldProgress->setTickPosition(QSlider::TicksBelow);
    sldProgress->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    sldProgress->setTracking(false);

    ////////////////////////////////////////////////////////////////////////////////
    // input data path
    txtDataPath[0] = new QLineEdit;
    txtDataPath[0]->setEnabled(false);

    QToolButton* btnBrowseVP0 = new QToolButton;
    btnBrowseVP0->setIcon(QIcon("icons/open.png"));
    connect(btnBrowseVP0, &QToolButton::clicked, this, &MainWindow::browseDataPathVP0);

    txtDataPath[1] = new QLineEdit;
    txtDataPath[1]->setEnabled(false);
    txtDataPath[1]->setVisible(false);

    btnBrowseVP1 = new QToolButton;
    btnBrowseVP1->setIcon(QIcon("icons/open.png"));
    connect(btnBrowseVP1, &QToolButton::clicked, this, &MainWindow::browseDataPathVP1);
    btnBrowseVP1->setVisible(false);

    QHBoxLayout* dataPathLayout = new QHBoxLayout;
    dataPathLayout->setSpacing(10);
    dataPathLayout->addWidget(txtDataPath[0], 1);
    dataPathLayout->addItem(new QSpacerItem(10, 20, QSizePolicy::Expanding,
                                            QSizePolicy::Minimum));
    dataPathLayout->addWidget(btnBrowseVP0);
    dataPathLayout->addItem(new QSpacerItem(10, 20, QSizePolicy::Expanding,
                                            QSizePolicy::Minimum));
    dataPathLayout->addWidget(txtDataPath[1], 1);
    dataPathLayout->addItem(new QSpacerItem(10, 20, QSizePolicy::Expanding,
                                            QSizePolicy::Minimum));
    dataPathLayout->addWidget(btnBrowseVP1);

    QGroupBox* dataPathGroup = new QGroupBox("Data Path");
    dataPathGroup->setLayout(dataPathLayout);

    ////////////////////////////////////////////////////////////////////////////////
    // output image path
    QCheckBox* chkEnableImageOutput = new QCheckBox("Export to Images");
    connect(chkEnableImageOutput, &QCheckBox::toggled, this,
            &MainWindow::enableImageOutput);

    txtOutputPath = new QLineEdit;
    txtOutputPath->setEnabled(false);

    QToolButton* btnBrowseOutput = new QToolButton;
    btnBrowseOutput->setIcon(QIcon("icons/save.png"));
    connect(btnBrowseOutput, &QToolButton::clicked, this, &MainWindow::browseOutputPath);
    connect(this, &MainWindow::outputPathChanged, renderer,
            & DualViewportRenderer::setImageOutputPath);

    QHBoxLayout* outputPathLayout = new QHBoxLayout;
    outputPathLayout->setSpacing(10);
    outputPathLayout->addWidget(chkEnableImageOutput);
    outputPathLayout->addItem(new QSpacerItem(10, 20, QSizePolicy::Expanding,
                                              QSizePolicy::Minimum));
    outputPathLayout->addWidget(txtOutputPath, 1);
    outputPathLayout->addItem(new QSpacerItem(10, 20, QSizePolicy::Expanding,
                                              QSizePolicy::Minimum));
    outputPathLayout->addWidget(btnBrowseOutput);

    QGroupBox* outputPathGroup = new QGroupBox("Image Output Path");
    outputPathGroup->setLayout(outputPathLayout);


    ////////////////////////////////////////////////////////////////////////////////
    // layout
    QHBoxLayout* inputOutputLayout = new QHBoxLayout;
    inputOutputLayout->addWidget(dataPathGroup);
    inputOutputLayout->addWidget(outputPathGroup);

    QVBoxLayout* viewLayout = new QVBoxLayout;
    viewLayout->addWidget(renderer, 2);
    viewLayout->addWidget(sldProgress);
    viewLayout->addLayout(inputOutputLayout);


    controller = new Controller(this);

    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addLayout(viewLayout);
    mainLayout->addWidget(controller);

    QWidget* mainWidget = new QWidget;
    mainWidget->setLayout(mainLayout);
    setCentralWidget(mainWidget);

    ////////////////////////////////////////////////////////////////////////////////
    // status
    lblStatusCurrentFrame = new QLabel(this);
    lblStatusCurrentFrame->setMargin(5);
    statusBar()->addPermanentWidget(lblStatusCurrentFrame, 2);

    lblStatusNumFrames = new QLabel(this);
    lblStatusNumFrames->setMargin(5);
    statusBar()->addPermanentWidget(lblStatusNumFrames, 2);


    lblStatusNumParticles = new QLabel(this);
    lblStatusNumParticles->setMargin(5);
    statusBar()->addPermanentWidget(lblStatusNumParticles, 5);

    statusBar()->addPermanentWidget(new QLabel, 3);
    statusBar()->setMinimumHeight(30);
    statusBar()->setSizeGripEnabled(false);

    resize(sizeHint());
}

//------------------------------------------------------------------------------------------
void MainWindow::setStatus(StatusType status_type, QString str)
{
    switch(status_type)
    {
    case STATUS_CURRENT_FRAME:
        lblStatusCurrentFrame->setText(str);
        break;

    case STATUS_NUM_FRAMES:
        lblStatusNumFrames->setText(str);
        break;

    case STATUS_NUM_PARTICLES:
        lblStatusNumParticles->setText(str);
        break;
    }

}

//------------------------------------------------------------------------------------------
void MainWindow::startVisualizeData(int viewport)
{
    txtDataPath[viewport]->setText(data_dirs_[viewport]);


    if(parameterLoader[viewport]->loadParams(data_dirs_[viewport]))
    {
        simParams_[viewport] = parameterLoader[viewport]->getSimulationParameters();

        updateStatusNumParticles(simParams_[viewport].num_sph_particle,
                                 simParams_[viewport].num_pd_particle,
                                 viewport);
        renderer->allocateMemory(simParams_[viewport].num_sph_particle,
                                 simParams_[viewport].num_pd_particle,
                                 simParams_[viewport].sph_particle_radius * simParams_[viewport].scaleFactor,
                                 simParams_[viewport].pd_particle_radius * simParams_[viewport].scaleFactor,
                                 viewport);

        controller->setSimulationInfo(parameterLoader[viewport]->getParametersAsString());

        dataReader->setSimulationDataInfo(data_dirs_,
          simParams_,
          num_frames_,
          num_viewports_);

    }
}

//------------------------------------------------------------------------------------------
void MainWindow::updateStatusNumParticles(int num_sph_particles, int num_pd_particles,
                                          int viewport)
{
    setStatus(STATUS_NUM_PARTICLES,
              QString("Particles: %1 (SPH: %2, Peridynamics: %3)")
              .arg(num_sph_particles + num_pd_particles).arg(num_sph_particles).arg(num_pd_particles));
}

//------------------------------------------------------------------------------------------
void MainWindow::keyPressEvent(QKeyEvent* e)
{
    switch(e->key())
    {
    case Qt::Key_Escape:
        close();
        break;

    case Qt::Key_B:
        browseDataPathVP0();
        break;

    case Qt::Key_D:
        controller->btnDualViewport->click();
        break;

    case Qt::Key_O:
        browseOutputPath();
        break;

    case Qt::Key_C:
        renderer->resetCameraPosition();
        break;

    case Qt::Key_R:
        controller->btnReverse->click();
        break;

    case Qt::Key_L:
    {
//        data_dirs_[0] = QString("/scratch/SPHP/DATA");
                    data_dirs_[0] = QString("/scratch/BunnyBreak/DATA");
        txtDataPath[0]->setText(data_dirs_[0]);

        if(parameterLoader[0]->loadParams(data_dirs_[0]))
        {
            startVisualizeData(0);
        }
    }
    break;

    case Qt::Key_Space:
        controller->btnPause->click();
        break;

    case Qt::Key_N:
        controller->btnNextFrame->click();
        break;

    case Qt::Key_X:
        controller->btnClipYZPlane->click();
        break;

    case Qt::Key_1:
        dataReader->resetToFirstFrame();
        break;

    default:
        renderer->keyPressEvent(e);
    }
}

//------------------------------------------------------------------------------------------
void MainWindow::keyReleaseEvent(QKeyEvent* _event)
{
    renderer->keyReleaseEvent(_event);
}

//------------------------------------------------------------------------------------------
void MainWindow::browseDataPathVP0()
{
    QString dataDir = QFileDialog::getExistingDirectory(this, tr("Select data directory"),
                                                        data_dirs_[0],
                                                        QFileDialog::ShowDirsOnly
                                                        | QFileDialog::DontResolveSymlinks);

    if(dataDir.trimmed() == "")
    {
        return;
    }

    data_dirs_[0] = dataDir;
    startVisualizeData(0);
}
//------------------------------------------------------------------------------------------

void MainWindow::browseDataPathVP1()
{
    QString dataDir = QFileDialog::getExistingDirectory(this, tr("Select data directory"),
                                                        data_dirs_[1],
                                                        QFileDialog::ShowDirsOnly
                                                        | QFileDialog::DontResolveSymlinks);

    if(dataDir.trimmed() == "")
    {
        return;
    }

    data_dirs_[1] = dataDir;
    startVisualizeData(1);
}

//-----------------------------------------------r-------------------------------------------
bool MainWindow::browseOutputPath()
{
    QString outputDir = QFileDialog::getExistingDirectory(this,
                                                          tr("Select image output directory"),
                                                          QDir::homePath(),
                                                          QFileDialog::ShowDirsOnly
                                                          | QFileDialog::DontResolveSymlinks);

    if(outputDir.trimmed() == "")
    {
        return false;
    }

    txtOutputPath->setText(outputDir);
    emit outputPathChanged(outputDir);
    return true;
}

//------------------------------------------------------------------------------------------
void MainWindow::enableImageOutput(bool status)
{
    controller->btnPause->click();

    if(status && txtOutputPath->text().trimmed().length() == 0)
    {
        if(browseOutputPath())
        {
            renderer->enableImageOutput(status);
        }
        else
        {
            QCheckBox* chkEnableOutput = qobject_cast<QCheckBox*>(QObject::sender());

            if(chkEnableOutput)
            {
                chkEnableOutput->setChecked(false);
            }
        }
    }
    else
    {
        renderer->enableImageOutput(status);
    }

    controller->btnPause->click();
}

//------------------------------------------------------------------------------------------
void MainWindow::updateStatusCurrentFrame(int current_frame)
{
    setStatus(STATUS_CURRENT_FRAME, QString("Rendering frame %1").arg(current_frame));
}

//------------------------------------------------------------------------------------------
void MainWindow::updateStatusNumFrames(int num_frames)
{
    setStatus(STATUS_NUM_FRAMES, QString("Total frames: %1").arg(num_frames));
}

//------------------------------------------------------------------------------------------
void MainWindow::updateNumFrames()
{
    if(num_viewports_ == 1)
    {
        num_frames_ = parameterLoader[0]->num_frames;
    }
    else
    {
        num_frames_ = (parameterLoader[0]->num_frames < parameterLoader[1]->num_frames) ?
                      parameterLoader[0]->num_frames : parameterLoader[1]->num_frames;
    }

    updateStatusNumFrames(num_frames_);
    sldProgress->setRange(0, num_frames_ - 1);
    dataReader->updateNumFrames(num_frames_);

}

//------------------------------------------------------------------------------------------
void MainWindow::enableDualViewport(bool status)
{
    txtDataPath[1]->setVisible(status);
    btnBrowseVP1->setVisible(status);

    if(status)
    {
        num_viewports_ = 2;
    }
    else
    {
        num_viewports_ = 1;
    }
}

