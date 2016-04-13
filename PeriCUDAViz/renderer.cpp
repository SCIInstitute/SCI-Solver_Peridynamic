//------------------------------------------------------------------------------------------
//
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------
#include "renderer.h"

Renderer::Renderer(QWidget* parent, int viewportID) :
    QOpenGLWidget(parent),
    bAllocatedMemory(false),
    bInitializedScene(false),
    bTextureAnisotropicFiltering(true),
//    enabledClipYZPlane(false),
    iboBackground(QOpenGLBuffer::IndexBuffer),
    iboGround(QOpenGLBuffer::IndexBuffer),
    iboBox(QOpenGLBuffer::IndexBuffer),
    cubeObject(NULL),
    planeObject(NULL),
    specialKeyPressed(Renderer::NO_KEY),
    mouseButtonPressed(Renderer::NO_BUTTON),
    currentFloorTexture(NO_FLOOR),
    currentEnvironmentTexture(NO_ENVIRONMENT_TEXTURE),
    translation_(0.0f, 0.0f, 0.0f),
    translationLag_(0.0f, 0.0f, 0.0f),
    rotation_(0.0f, 0.0f, 0.0f),
    rotationLag_(0.0f, 0.0f, 0.0f),
    zooming_(0.0f),
    cameraPosition_(DEFAULT_CAMERA_POSITION),
    cameraFocus_(DEFAULT_CAMERA_FOCUS),
    cameraUpDirection_(0.0f, 1.0f, 0.0f),
    currentMouseTransTarget(TRANSFORM_CAMERA),
    currentParticleViewMode(SPHERES_VIEW),
    currentParticleColorMode(COLOR_PARTICLE_TYPE),
    imageOutputPath(""),
    enabledImageOutput(false),
    pausedImageOutput(false),
    bHideInvisibleParticles(false),
    outputImage(NULL),
    viewportID_(viewportID)
{
    retinaScale = devicePixelRatio();;

    isParticlesReady = false;

    num_sph_particles_ = 0;
    num_pd_particles_ = 0;
    num_particles_ = 0;

    particleRandomColors = NULL;
    particleRampColors = NULL;
    particleSimulationColors = NULL;

}


//------------------------------------------------------------------------------------------
QSize Renderer::sizeHint() const
{
    return QSize(1800, 1000);
}

//------------------------------------------------------------------------------------------
QSize Renderer::minimumSizeHint() const
{
    return QSize(50, 50);
}

//------------------------------------------------------------------------------------------
void Renderer::mousePressEvent(QMouseEvent* _event)
{
    lastMousePos = QVector2D(_event->localPos());

    if(_event->button() == Qt::RightButton)
    {
        mouseButtonPressed = RIGHT_BUTTON;
    }
    else
    {
        mouseButtonPressed = LEFT_BUTTON;
    }
}

//------------------------------------------------------------------------------------------
void Renderer::mouseMoveEvent(QMouseEvent* _event)
{
    QVector2D mouseMoved = QVector2D(_event->localPos()) - lastMousePos;

    switch(specialKeyPressed)
    {
    case Renderer::NO_KEY:
    {

        if(mouseButtonPressed == RIGHT_BUTTON)
        {
            translation_.setX(translation_.x() + mouseMoved.x() / 50.0f);
            translation_.setY(translation_.y() - mouseMoved.y() / 50.0f);
        }
        else
        {
            rotation_.setX(rotation_.x() - mouseMoved.x() / 5.0f);
            rotation_.setY(rotation_.y() - mouseMoved.y() / 5.0f);

            QPointF center = QPointF(0.5 * width(), 0.5 * height());
            QPointF escentricity = _event->localPos() - center;
            escentricity.setX(escentricity.x() / center.x());
            escentricity.setY(escentricity.y() / center.y());
            rotation_.setZ(rotation_.z() - (mouseMoved.x()*escentricity.y() - mouseMoved.y() *
                                            escentricity.x()) / 5.0f);

        }

    }
    break;

    case Renderer::SHIFT_KEY:
    {
        if(mouseButtonPressed == RIGHT_BUTTON)
        {
            QVector2D dir = mouseMoved.normalized();
            zooming_ += mouseMoved.length() * dir.x() / 500.0f;

        }
        else
        {
            rotation_.setX(rotation_.x() + mouseMoved.x() / 5.0f);
            rotation_.setZ(rotation_.z() + mouseMoved.y() / 5.0f);

        }
    }
    break;

    case Renderer::CTRL_KEY:
        break;
    }

    lastMousePos = QVector2D(_event->localPos());
    update();
}

//------------------------------------------------------------------------------------------
void Renderer::mouseReleaseEvent(QWheelEvent* _event)
{
    mouseButtonPressed = NO_BUTTON;
}

//------------------------------------------------------------------------------------------
void Renderer::wheelEvent(QWheelEvent* _event)
{
    if(!_event->angleDelta().isNull())
    {
        zooming_ +=  (_event->angleDelta().x() + _event->angleDelta().y()) / 500.0f;

    }

    update();
}

//------------------------------------------------------------------------------------------
void Renderer::allocateMemory(int num_sph_particles, int num_pd_particles,
                              float sph_particle_radius, float pd_particle_radius)
{
    if(!isValid())
    {
        return;
    }

    makeCurrent();

    num_sph_particles_ = num_sph_particles;
    num_pd_particles_ = num_pd_particles;
    num_particles_ = num_sph_particles + num_pd_particles;
    sph_particle_radius_ = sph_particle_radius;
    pd_particle_radius_ = pd_particle_radius;

//    qDebug() << num_sph_particles << num_pd_particles;
//    qDebug() << sph_particle_radius << pd_particle_radius;

    if(vboParticles.isCreated())
    {
        vboParticles.destroy();
    }

    vboParticles.create();
    vboParticles.bind();
    vboParticles.allocate(num_particles_ * (sizeof(real_t) * (4 +
                                                              3) // 4: coordinate, 3: color
                                            + sizeof(int))); // activity variable
    vboParticles.release();

    initParticlesRandomColors();
    initParticlesRampColors();

    if(particleSimulationColors)
    {
        delete[] particleSimulationColors;
    }

    particleSimulationColors = new GLREAL[num_particles_ * 3];

    setParticleColorMode(currentParticleColorMode);
    initParticlesVAO(PROGRAM_POINT_SPHERE_VIEW);
//    initParticlesVAO(PROGRAM_SURFACE_VIEW);
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::setSPHParticleColor(float r, float g, float b)
{
    if(!isValid())
    {
        return;
    }

    pointSphereSPHMaterial.setDiffuse(QVector4D(r, g, b, 1.0f));
    makeCurrent();
    glBindBuffer(GL_UNIFORM_BUFFER, UBOPointSphereSPHMaterial);
    glBufferData(GL_UNIFORM_BUFFER, pointSphereSPHMaterial.getStructSize(),
                 &pointSphereSPHMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::setPDParticleColor(float _r, float g, float b)
{
    if(!isValid())
    {
        return;
    }

    pointSpherePDMaterial.setDiffuse(QVector4D(_r, g, b, 1.0f));
    makeCurrent();
    glBindBuffer(GL_UNIFORM_BUFFER, UBOPointSpherePDMaterial);
    glBufferData(GL_UNIFORM_BUFFER, pointSpherePDMaterial.getStructSize(),
                 &pointSpherePDMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    doneCurrent();
}
//------------------------------------------------------------------------------------------

void Renderer::setSurfaceDiffuseColor(float r, float g, float b)
{
    if(!isValid())
    {
        return;
    }

    surfaceMaterial.setDiffuse(QVector4D(r, g, b, 1.0f));
    makeCurrent();
    glBindBuffer(GL_UNIFORM_BUFFER, UBOSurfaceMaterial);
    glBufferData(GL_UNIFORM_BUFFER, surfaceMaterial.getStructSize(),
                 NULL, GL_STREAM_DRAW);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, surfaceMaterial.getStructSize(),
                    &surfaceMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::keyPressEvent(QKeyEvent* _event)
{
    switch(_event->key())
    {
    case Qt::Key_Shift:
        specialKeyPressed = Renderer::SHIFT_KEY;
        break;

    case Qt::Key_Plus:
    {
        zooming_ -= 0.1f;
    }
    break;

    case Qt::Key_Minus:
    {
        zooming_ += 0.1f;
    }
    break;

    case Qt::Key_Up:
    {
        translation_ += QVector3D(0.0f, 0.5f, 0.0f);
    }
    break;

    case Qt::Key_Down:
    {
        translation_ -= QVector3D(0.0f, 0.5f, 0.0f);
    }
    break;

    case Qt::Key_Left:
    {
        translation_ -= QVector3D(0.5f, 0.0f, 0.0f);
    }
    break;

    case Qt::Key_Right:
    {
        translation_ += QVector3D(0.5f, 0.0f, 0.0f);

    }
    break;

//    case Qt::Key_X:
//        rotation-= QVector3D(0.0f, 3.0f, 0.0f);
//        break;

    default:
        QOpenGLWidget::keyPressEvent(_event);
    }
}

//------------------------------------------------------------------------------------------
void Renderer::keyReleaseEvent(QKeyEvent*)
{
    specialKeyPressed = Renderer::NO_KEY;
}

//------------------------------------------------------------------------------------------
void Renderer::initializeGL()
{
    initializeOpenGLFunctions();

    glEnable (GL_DEPTH_TEST);
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    checkOpenGLVersion();

    if(!bInitializedScene)
    {
        initScene();
        bInitializedScene = true;
    }
}

//------------------------------------------------------------------------------------------
void Renderer::resizeGL(int w, int h)
{
    projectionMatrix.setToIdentity();
    projectionMatrix.perspective(45, (float)w / (float)h, 0.1f, 10000.0f);

    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 3 * SIZE_OF_MAT4, SIZE_OF_MAT4,
                    projectionMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // prepare output image
    if(outputImage)
    {
        delete outputImage;
    }

    outputImage = new QImage(w, h, QImage::Format_ARGB32);
}

//------------------------------------------------------------------------------------------
void Renderer::paintGL()
{
    if(!bInitializedScene)
    {
        return;
    }

    QVector3D cameraPosOld = cameraPosition_;
    QVector3D cameraFocusOld = cameraFocus_;

    switch(currentMouseTransTarget)
    {
    case TRANSFORM_CAMERA:
    {
        translateCamera();
        rotateCamera();
    }
    break;

    case TRANSFORM_LIGHT:
    {
        translateLight();
    }
    break;
    }

    updateCamera();

    float camDiff = (cameraPosOld - cameraPosition_).lengthSquared() +
                    (cameraFocusOld - cameraFocus_).lengthSquared();

    if(camDiff > 1e-5)
    {
        emit cameraChanged(cameraPosition_, cameraFocus_, cameraUpDirection_);
    }

    // render scene
    renderScene();
}

//------------------------------------------------------------------------------------------
void Renderer::setCamera(QVector3D cameraPos, QVector3D cameraFocus,
                         QVector3D cameraUpDir)
{
    cameraPosition_ = cameraPos;
    cameraFocus_ = cameraFocus;
    cameraUpDirection_ = cameraUpDir;
}

//------------------------------------------------------------------------------------------
void Renderer::enableAnisotropicTextureFiltering(bool status)
{
    bTextureAnisotropicFiltering = status;
}

void Renderer::enableClipYZPlane(bool status)
{
//    enabledClipYZPlane = _status;
    if(!isValid())
    {
        return;
    }

    makeCurrent();

    if (status)
    {
        glEnable (GL_CLIP_PLANE0);
    }
    else
    {
        glDisable (GL_CLIP_PLANE0);
    }

    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::hideInvisibleParticles(bool status)
{
    bHideInvisibleParticles = status;
}

//------------------------------------------------------------------------------------------
void Renderer::setParticleViewMode(int view_mode)
{
    currentParticleViewMode = static_cast<ParticleViewMode>(view_mode);
}

//------------------------------------------------------------------------------------------
void Renderer::setParticleColorMode(int color_mode)
{
    currentParticleColorMode = static_cast<ParticleColorMode>(color_mode);

    if(!vboParticles.isCreated())
    {
        qDebug() << "vaoParticles is not created!";
        return;
    }

    GLREAL* colorArray;

    switch(color_mode)
    {
    case COLOR_RANDOM:
        colorArray = particleRandomColors;
        break;

    case COLOR_RAMP:
        colorArray = particleRampColors;
        break;

    default:
        return;
    }

    vboParticles.bind();
    vboParticles.write(4 * num_particles_ * sizeof(GLREAL), colorArray,
                       3 * num_particles_ * sizeof(GLREAL));
    vboParticles.release();
}

//------------------------------------------------------------------------------------------
void Renderer::setParticlePositions(real_t* pd_positions, real_t* sph_positions,
                                    int* pd_activities, int* sph_activities,
                                    int current_frame)
{
    if(!isValid())
    {
        return;
    }

    if(!vboParticles.isCreated())
    {
        PRINT_ERROR("vboParticle is not created!")
        return;
    }


    makeCurrent();
    current_frame_ = current_frame;

    vboParticles.bind();
    vboParticles.write(0, pd_positions,
                       num_pd_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(num_pd_particles_ * 4 * sizeof(GLREAL),
                       sph_positions,
                       num_sph_particles_ * 4 * sizeof(GLREAL));

    if(bHideInvisibleParticles)
    {
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3), pd_activities,
                           num_pd_particles_ * sizeof(GLint));
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3) +
                           num_pd_particles_ * sizeof(GLint),
                           sph_activities,
                           num_sph_particles_ * sizeof(GLint));

//        int max_print = num_pd_particles_ < 100 ? num_pd_particles_ : 100;
//        qDebug() << "max_print: " << max_print;

//        for(int i = 0; i < max_print; ++i)
//        {
//            qDebug() << i << " " << pd_activities[i];
//        }
    }

    vboParticles.release();

//    initParticlesVAO(PROGRAM_POINT_SPHERE_VIEW);
    isParticlesReady = true;
    doneCurrent();
//    qDebug() << "Num sph: " << num_sph_particles_;





#if 0

        {
            QFile fOutPD(QString("/Users/nghia/GoogleDrive/Research/Data/Txt/frame.%1.pd.txt").arg(
                             current_frame_));

            if (fOutPD.open(QFile::WriteOnly | QFile::Text))
            {
                QTextStream s(&fOutPD);

                s << "particle_radius " << pd_particle_radius_ << '\n';

                for (int i = 0; i < num_pd_particles_; ++i)
                {
                    s <<  pd_positions[i * 4] << " " <<  pd_positions[i * 4 + 1] << " " <<
                      pd_positions[i * 4 + 2] << '\n';
                }
            }
            else
            {
                qDebug() << "error opening output file";
                exit(EXIT_FAILURE);
            }

            fOutPD.close();


            QFile fOutSPH(QString("/Users/nghia/GoogleDrive/Research/Data/Txt/frame.%1.sph.txt").arg(
                              current_frame_));

            if (fOutSPH.open(QFile::WriteOnly | QFile::Text))
            {
                QTextStream s(&fOutSPH);

                s << "particle_radius " << sph_particle_radius_ << '\n';

                for (int i = 0; i < num_sph_particles_; ++i)
                {
                    s <<  sph_positions[i * 4] << " " <<  sph_positions[i * 4 + 1] << " " <<
                      sph_positions[i * 4 + 2] << '\n';
                }
            }
            else
            {
                qDebug() << "error opening output file";
                exit(EXIT_FAILURE);
            }

            fOutSPH.close();

            qDebug() << "Frame written: " << current_frame_;
        }

#endif

#if 0
        ////////////////////////////////////////////////////////////////////////////////
        {
            QFile fOutPD(QString("/Users/nghia/OBJ/frame.%1.pd.txt").arg(_currentFrame));

            if (fOutPD.open(QFile::WriteOnly | QFile::Text))
            {
                QTextStream s(&fOutPD);

//            s << "particle_radius " << PeridynamicsParticleRadius << '\n';

                for (int i = 0; i < numPeridynamicsParticles; ++i)
                {
                    s <<  "v " << _pdParticles[i * 4] << " " <<  _pdParticles[i * 4 + 1] << " " <<
                      _pdParticles[i * 4 + 2] << '\n';
                }
            }
            else
            {
                qDebug() << "error opening output file";
                exit(EXIT_FAILURE);
            }

            fOutPD.close();


            QFile fOutSPH(QString("/Users/nghia/OBJ/frame.%1.sph.txt").arg(_currentFrame));

            if (fOutSPH.open(QFile::WriteOnly | QFile::Text))
            {
                QTextStream s(&fOutSPH);

//                s << "particle_radius " << SPHParticleRadius << '\n';

                for (int i = numPeridynamicsParticles; i < numTotalParticles; ++i)
                {
                    s << "v " <<  _pdParticles[i * 4] << " " <<  _pdParticles[i * 4 + 1] << " " <<
                      _pdParticles[i * 4 + 2] << '\n';
                }
            }
            else
            {
                qDebug() << "error opening output file";
                exit(EXIT_FAILURE);
            }

            fOutSPH.close();
        }

        qDebug() << "Files written.";


#endif

}

//------------------------------------------------------------------------------------------
void Renderer::setParticleDensitiesPositions(real_t* _pdParticles, real_t* _sphParticles,
                                             int* pd_activities, int* sph_activities,
                                             real_t* sph_densities, int current_frame)
{
    if(!isValid())
    {
        return;
    }

    if(!vboParticles.isCreated())
    {
        PRINT_ERROR("vboParticle is not created!")
        return;
    }

    // find the max and min of density
    real_t maxDensity = -1e10;
    real_t minDensity = 1e10;

#pragma unroll 8

    for(int i = 0; i < num_sph_particles_; ++i)
    {
        if(maxDensity < sph_densities[i])
        {
            maxDensity = sph_densities[i];
        }

        if(minDensity > sph_densities[i])
        {
            minDensity = sph_densities[i];
        }

//        qDebug() << _densities[i];
    }

//    qDebug() << maxDensity << minDensity;
//        for(int i = 0; i < numSPHParticles; ++i)
//        {
//            qDebug() << _sphParticles[i * 4] << _sphParticles[i * 4 + 1] << _sphParticles[i * 4 + 2];
//        }



    real_t scale = maxDensity - minDensity;

    if(scale == 0.0)
    {
        scale = 1.0;
    }

    real_t t;

#pragma unroll 8

    for(int i = 0; i < num_pd_particles_; ++i)
    {
        t = 0;
        colorRamp(t, &(particleSimulationColors)[i * 3]);
    }

    for(int i = 0; i < num_sph_particles_; ++i)
    {
        t = (sph_densities[i] - minDensity) / scale;
        colorRamp(t, &(particleSimulationColors)[(i + num_pd_particles_) *
                                                 3]);
    }


    makeCurrent();
    current_frame_ = current_frame;
    vboParticles.bind();
    vboParticles.write(0, _pdParticles,
                       num_pd_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(num_pd_particles_ * 4 * sizeof(GLREAL),
                       _sphParticles,
                       num_sph_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(4 * num_particles_ * sizeof(GLREAL),
                       particleSimulationColors,
                       3 * num_particles_ * sizeof(GLREAL));

    if(bHideInvisibleParticles)
    {
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3), pd_activities,
                           num_pd_particles_ * sizeof(GLint));
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3) +
                           num_pd_particles_ * sizeof(GLint),
                           sph_activities,
                           num_sph_particles_ * sizeof(GLint));
    }

    vboParticles.release();
    isParticlesReady = true;
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::setParticleStiffnesPositions(real_t* pd_positions, real_t* sph_positions,
                                            int* pd_activities, int* sph_activities,
                                            real_t* pd_stiffness, int current_frame)
{
    if(!isValid())
    {
        return;
    }

    if(!vboParticles.isCreated())
    {
        PRINT_ERROR("vboParticle is not created!")
        return;
    }

    // find the max and min of density
    real_t maxStiff = -1e100;
    real_t minStiff = 1e100;

#pragma unroll 8

    for(int i = 0; i < num_pd_particles_; ++i)
    {
        if(maxStiff < pd_stiffness[i])
        {
            maxStiff = pd_stiffness[i];
        }

        if(minStiff > pd_stiffness[i])
        {
            minStiff = pd_stiffness[i];
        }

//        qDebug() << _particleStiffness[i];

    }

//    qDebug() << minStiff<< maxStiff;
//qDebug() << numPeridynamicsParticles;

    real_t scale = (real_t)(maxStiff - minStiff);

    if(scale == 0.0)
    {
        scale = 1.0;
    }

    real_t t;

#pragma unroll 8

    for(int i = 0; i < num_pd_particles_; ++i)
    {
//        qDebug() << _particleStretches[i];
        t = (real_t)(pd_stiffness[i] - minStiff) / scale;

//        if(t < 0.0 || t > 1.0)
//        {
//            t = 0.99;
//        }

        colorRamp(t, &(particleSimulationColors)[i * 3]);
    }

    for(int i = num_pd_particles_; i < num_particles_; ++i)
    {
        colorRamp(0.2, &(particleSimulationColors)[i * 3]);
    }

    makeCurrent();
    current_frame_ = current_frame;

    vboParticles.bind();
    vboParticles.write(0, pd_positions,
                       num_pd_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(num_pd_particles_ * 4 * sizeof(GLREAL),
                       sph_positions,
                       num_sph_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(4 * num_particles_ * sizeof(GLREAL),
                       particleSimulationColors,
                       3 * num_particles_ * sizeof(GLREAL));

    if(bHideInvisibleParticles)
    {
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3), pd_activities,
                           num_pd_particles_ * sizeof(GLint));
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3) +
                           num_pd_particles_ * sizeof(GLint),
                           sph_activities,
                           num_sph_particles_ * sizeof(GLint));
    }

    vboParticles.release();

    isParticlesReady = true;
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::setParticleActivitiesPositions(real_t* pd_positions, real_t* sph_positions,
                                              int* pd_activities,
                                              int* sph_activities, int current_frame)
{
    if(!isValid())
    {
        return;
    }

    if(!vboParticles.isCreated())
    {
        PRINT_ERROR("vboParticle is not created!")
        return;
    }

    // find the max and min of density
#pragma unroll 8

    for(int i = 0; i < num_pd_particles_; ++i)
    {
        colorRamp((real_t)(pd_activities[i]) / (real_t)NUM_ACTIVITY_MODE,
                  &(particleSimulationColors)[i * 3]);
    }

    for(int i = 0; i < num_sph_particles_; ++i)
    {
        colorRamp((real_t)(sph_activities[i]) / (real_t)NUM_ACTIVITY_MODE,
                  &(particleSimulationColors)[(i + num_pd_particles_) * 3]);
    }

    makeCurrent();
    current_frame_ = current_frame;

    vboParticles.bind();
    vboParticles.write(0, pd_positions,
                       num_pd_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(num_pd_particles_ * 4 * sizeof(GLREAL),
                       sph_positions,
                       num_sph_particles_ * 4 * sizeof(GLREAL));
    vboParticles.write(4 * num_particles_ * sizeof(GLREAL),
                       particleSimulationColors,
                       3 * num_particles_ * sizeof(GLREAL));

    if(bHideInvisibleParticles)
    {
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3), pd_activities,
                           num_pd_particles_ * sizeof(GLint));
        vboParticles.write(num_particles_ * sizeof(GLREAL) * (4 + 3) +
                           num_pd_particles_ * sizeof(GLint),
                           sph_activities,
                           num_sph_particles_ * sizeof(GLint));
    }

    vboParticles.release();

    isParticlesReady = true;
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::setEnvironmentTexture(int texture)
{
    currentEnvironmentTexture = static_cast<EnvironmentTexture>(texture);
}

//------------------------------------------------------------------------------------------
void Renderer::setFloorTexture(int texture)
{
    currentFloorTexture = static_cast<FloorTexture>(texture);
}

//------------------------------------------------------------------------------------------
void Renderer::setMouseTransformationTarget(MouseTransformationTarget mouse_target)
{
    currentMouseTransTarget = mouse_target;
}

//------------------------------------------------------------------------------------------
void Renderer::setLightIntensity(int intensity)
{
    if(!isValid())
    {
        return;
    }

    makeCurrent();
    light.intensity = (GLfloat)intensity / 100.0f;
    glBindBuffer(GL_UNIFORM_BUFFER, UBOLight);
    glBufferData(GL_UNIFORM_BUFFER, light.getStructSize(),
                 &light, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    doneCurrent();
    update();
}

//------------------------------------------------------------------------------------------
void Renderer::resetCameraPosition()
{
    cameraPosition_ = DEFAULT_CAMERA_POSITION;
    cameraFocus_ = DEFAULT_CAMERA_FOCUS;
    cameraUpDirection_ = QVector3D(0.0f, 1.0f, 0.0f);

    update();
}

//------------------------------------------------------------------------------------------
void Renderer::resetLightPosition()
{
    makeCurrent();
    light.position = DEFAULT_LIGHT_DIRECTION;
    glBindBuffer(GL_UNIFORM_BUFFER, UBOLight);
    glBufferData(GL_UNIFORM_BUFFER, light.getStructSize(),
                 &light, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    doneCurrent();
}

//------------------------------------------------------------------------------------------
void Renderer::enableImageOutput(bool status)
{
    enabledImageOutput = status;
}

//------------------------------------------------------------------------------------------
void Renderer::pauseImageOutput(bool status)
{
    pausedImageOutput = status;
}

//------------------------------------------------------------------------------------------
void Renderer::setImageOutputPath(QString output_path)
{
    imageOutputPath = output_path;
}

//------------------------------------------------------------------------------------------
void Renderer::checkOpenGLVersion()
{
    QString verStr = QString((const char*)glGetString(GL_VERSION));
    int major = verStr.left(verStr.indexOf(".")).toInt();
    int minor = verStr.mid(verStr.indexOf(".") + 1, 1).toInt();

    if(!(major >= 4 && minor >= 1))
    {
        QMessageBox::critical(this, "Error",
                              QString("Your OpenGL version is %1.%2, which does not satisfy this program requirement (OpenGL >= 4.1)")
                              .arg(major).arg(minor));
        close();
    }

//    qDebug() << major << minor;
//    qDebug() << verStr;
    //    TRUE_OR_DIE(major >= 4 && minor >= 1, "OpenGL version must >= 4.1");
}

//------------------------------------------------------------------------------------------
void Renderer::initScene()
{
    initShaderPrograms();
    initTexture();
    initSceneMemory();
    initVertexArrayObjects();
    initSharedBlockUniform();
    initSceneMatrices();
    initFrameBufferObject();

    glEnable(GL_DEPTH_TEST);
}

//------------------------------------------------------------------------------------------
void Renderer::initShaderPrograms()
{
    vertexShaderSourceMap.insert(PROGRAM_POINT_SPHERE_VIEW,
                                 ":/shaders/point-sphere-view.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_SURFACE_VIEW, ":/shaders/surface-view.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_RENDER_BACKGROUND,
                                 ":/shaders/background.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_RENDER_GROUND,
                                 ":/shaders/ground.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_RENDER_BOX,
                                 ":/shaders/box.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_RENDER_LIGHT,
                                 ":/shaders/light.vs.glsl");
    vertexShaderSourceMap.insert(PROGRAM_RENDER_DEPTH_BUFFER,
                                 ":/shaders/depth-map.vs.glsl");

    fragmentShaderSourceMap.insert(PROGRAM_POINT_SPHERE_VIEW,
                                   ":/shaders/point-sphere-view.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_SURFACE_VIEW, ":/shaders/surface-view.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_RENDER_BACKGROUND,
                                   ":/shaders/background.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_RENDER_GROUND,
                                   ":/shaders/ground.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_RENDER_BOX,
                                   ":/shaders/box.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_RENDER_LIGHT,
                                   ":/shaders/light.fs.glsl");
    fragmentShaderSourceMap.insert(PROGRAM_RENDER_DEPTH_BUFFER,
                                   ":/shaders/depth-map.fs.glsl");

//    for(int i = 0; i < NUM_GLSL_PROGRAMS; ++i)
//    {
//        TRUE_OR_DIE(initProgram(static_cast<GLSLPrograms>(i)), "Cannot init program");
//    }

    TRUE_OR_DIE(initRenderBackgroundProgram(), "Cannot init program drawing background");
    TRUE_OR_DIE(initRenderGroundProgram(), "Cannot init program drawing ground");
    TRUE_OR_DIE(initRenderBoxProgram(), "Cannot init program drawing box");
    TRUE_OR_DIE(initRenderLightProgram(), "Cannot init program drawing light");
    TRUE_OR_DIE(initRenderDepthBufferProgram(), "Cannot init program depth-buffer");
    TRUE_OR_DIE(initPointSphereViewProgram(), "Cannot init program point-sphere-view");

}

//------------------------------------------------------------------------------------------
bool Renderer::validateShaderPrograms(GLSLPrograms program)
{
    GLint status;
    GLint logLen;
    GLchar log[1024];

    glValidateProgram(glslPrograms[program]->programId());
    glGetProgramiv(glslPrograms[program]->programId(), GL_VALIDATE_STATUS, &status);

    glGetProgramiv(glslPrograms[program]->programId(), GL_INFO_LOG_LENGTH, &logLen);

    if(logLen > 0)
    {
        glGetProgramInfoLog(glslPrograms[program]->programId(), logLen, &logLen, log);

        if(QString(log).trimmed().length() != 0)
        {
            qDebug() << "ShadingMode: " << program << ", log: " << log;
        }
    }

    return (status == GL_TRUE);
}

//------------------------------------------------------------------------------------------
bool Renderer::initSurfaceViewProgram()
{
    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initPointSphereViewProgram()
{

    GLint location;
    glslPrograms[PROGRAM_POINT_SPHERE_VIEW] = new QOpenGLShaderProgram;
    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_POINT_SPHERE_VIEW];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_POINT_SPHERE_VIEW));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_POINT_SPHERE_VIEW));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = program->attributeLocation("v_coord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrVertex[PROGRAM_POINT_SPHERE_VIEW] = location;;

    location = program->attributeLocation("v_activity");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute activity.");
    attrActivity[PROGRAM_POINT_SPHERE_VIEW] = location;

    location = program->attributeLocation("v_color");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrColor[PROGRAM_POINT_SPHERE_VIEW] = location;

    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_POINT_SPHERE_VIEW] = location;

    location = glGetUniformBlockIndex(program->programId(), "Light");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniLight[PROGRAM_POINT_SPHERE_VIEW] = location;

    location = glGetUniformBlockIndex(program->programId(), "Material");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMaterial[PROGRAM_POINT_SPHERE_VIEW] = location;

    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initRenderBackgroundProgram()
{
    GLint location;
    glslPrograms[PROGRAM_RENDER_BACKGROUND] = new QOpenGLShaderProgram;
    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_BACKGROUND];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_RENDER_BACKGROUND));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_RENDER_BACKGROUND));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = program->attributeLocation("v_coord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrVertex[PROGRAM_RENDER_BACKGROUND] = location;

    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_RENDER_BACKGROUND] = location;

    location = program->uniformLocation("cameraPosition");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform cameraPosition.");
    uniCameraPosition[PROGRAM_RENDER_BACKGROUND] = location;

    location = program->uniformLocation("envTex");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform envTex.");
    uniObjTexture[PROGRAM_RENDER_BACKGROUND] = location;

    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initRenderGroundProgram()
{
    QOpenGLShaderProgram* program;
    GLint location;

    /////////////////////////////////////////////////////////////////
    glslPrograms[PROGRAM_RENDER_GROUND] = new QOpenGLShaderProgram;
    program = glslPrograms[PROGRAM_RENDER_GROUND];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_RENDER_GROUND));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_RENDER_GROUND));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = program->attributeLocation("v_coord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrVertex[PROGRAM_RENDER_GROUND] = location;

    location = program->attributeLocation("v_normal");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex normal.");
    attrNormal[PROGRAM_RENDER_GROUND] = location;

    location = program->attributeLocation("v_texCoord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute texture coordinate.");
    attrTexCoord[PROGRAM_RENDER_GROUND] = location;


    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_RENDER_GROUND] = location;


    location = glGetUniformBlockIndex(program->programId(), "Light");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniLight[PROGRAM_RENDER_GROUND] = location;

    location = glGetUniformBlockIndex(program->programId(), "Material");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMaterial[PROGRAM_RENDER_GROUND] = location;

    location = program->uniformLocation("cameraPosition");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform cameraPosition.");
    uniCameraPosition[PROGRAM_RENDER_GROUND] = location;

    location = program->uniformLocation("objTex");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform objTex.");
    uniObjTexture[PROGRAM_RENDER_GROUND] = location;

    location = program->uniformLocation("hasObjTex");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform hasObjTex.");
    uniHasObjTexture[PROGRAM_RENDER_GROUND] = location;

    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initRenderBoxProgram()
{
    QOpenGLShaderProgram* program;
    GLint location;

    /////////////////////////////////////////////////////////////////
    glslPrograms[PROGRAM_RENDER_BOX] = new QOpenGLShaderProgram;
    program = glslPrograms[PROGRAM_RENDER_BOX];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_RENDER_BOX));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_RENDER_BOX));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = program->attributeLocation("v_coord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrVertex[PROGRAM_RENDER_BOX] = location;

    location = program->attributeLocation("v_normal");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex normal.");
    attrNormal[PROGRAM_RENDER_BOX] = location;

    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_RENDER_BOX] = location;

    location = glGetUniformBlockIndex(program->programId(), "Light");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniLight[PROGRAM_RENDER_BOX] = location;

    location = glGetUniformBlockIndex(program->programId(), "Material");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMaterial[PROGRAM_RENDER_BOX] = location;

    location = program->uniformLocation("cameraPosition");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform cameraPosition.");
    uniCameraPosition[PROGRAM_RENDER_BOX] = location;

    location = program->uniformLocation("objTex");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform objTex.");
    uniObjTexture[PROGRAM_RENDER_BOX] = location;

    location = program->uniformLocation("hasObjTex");
    TRUE_OR_DIE(location >= 0, "Cannot bind uniform hasObjTex.");
    uniHasObjTexture[PROGRAM_RENDER_BOX] = location;

    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initRenderLightProgram()
{
    GLint location;
    glslPrograms[PROGRAM_RENDER_LIGHT] = new QOpenGLShaderProgram;
    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_LIGHT];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_RENDER_LIGHT));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_RENDER_LIGHT));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_RENDER_LIGHT] = location;

    location = glGetUniformBlockIndex(program->programId(), "Light");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniLight[PROGRAM_RENDER_LIGHT] = location;

    return true;
}

//------------------------------------------------------------------------------------------
bool Renderer::initRenderDepthBufferProgram()
{
    GLint location;
    glslPrograms[PROGRAM_RENDER_DEPTH_BUFFER] = new QOpenGLShaderProgram;
    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_DEPTH_BUFFER];
    bool success;

    success = program->addShaderFromSourceFile(QOpenGLShader::Vertex,
                                               vertexShaderSourceMap.value(PROGRAM_RENDER_DEPTH_BUFFER));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                               fragmentShaderSourceMap.value(PROGRAM_RENDER_DEPTH_BUFFER));
    TRUE_OR_DIE(success, "Cannot compile shader from file.");

    success = program->link();
    TRUE_OR_DIE(success, "Cannot link GLSL program.");

    location = program->attributeLocation("v_coord");
    TRUE_OR_DIE(location >= 0, "Cannot bind attribute vertex coordinate.");
    attrVertex[PROGRAM_RENDER_DEPTH_BUFFER] = location;

    location = glGetUniformBlockIndex(program->programId(), "Matrices");
    TRUE_OR_DIE(location >= 0, "Cannot bind block uniform.");
    uniMatrices[PROGRAM_RENDER_DEPTH_BUFFER] = location;


    return true;
}

//------------------------------------------------------------------------------------------
void Renderer::initSceneMatrices()
{


    /////////////////////////////////////////////////////////////////
    // background
    backgroundCubeModelMatrix.setToIdentity();
    backgroundCubeModelMatrix.scale(1000.0f);

    /////////////////////////////////////////////////////////////////
    // ground
    groundModelMatrix.setToIdentity();
    groundModelMatrix.translate(DEFAULT_GROUND_POSITION);
    groundModelMatrix.scale(GROUND_PLANE_SIZE);
    groundNormalMatrix = QMatrix4x4(groundModelMatrix.normalMatrix());

    /////////////////////////////////////////////////////////////////
    // box
    boxModelMatrix.setToIdentity();
    boxModelMatrix.scale(1, 1, 2);
    boxModelMatrix.translate(DEFAULT_BOX_POSITION);
    boxNormalMatrix = QMatrix4x4(boxModelMatrix.normalMatrix());

    /////////////////////////////////////////////////////////////////
    // particles
    particlesModelMatrix.setToIdentity();
//    particlesModelMatrix.translate(DEFAULT_BOX_POSITION);
}

//------------------------------------------------------------------------------------------
void Renderer::initSharedBlockUniform()
{
    /////////////////////////////////////////////////////////////////
    // setup the light and material
    light.position = DEFAULT_LIGHT_DIRECTION;
    light.intensity = 0.8f;


    groundMaterial.setDiffuse(QVector4D(-1.0f, 0.45f, 1.0f, 1.0f));
    groundMaterial.setSpecular(QVector4D(0.5f, 0.5f, 0.5f, 1.0f));
    groundMaterial.shininess = 150.0f;

    boxMaterial.setDiffuse(QVector4D(0.4f, 0.5f, 0.0f, 1.0f));
    boxMaterial.setSpecular(QVector4D(0.5f, 0.5f, 0.5f, 1.0f));
    boxMaterial.shininess = 150.0f;

    surfaceMaterial.setDiffuse(QVector4D(0.02f, 0.45f, 1.0f, 1.0f));
    surfaceMaterial.setSpecular(QVector4D(0.5f, 0.5f, 0.5f, 1.0f));
    surfaceMaterial.shininess = 150.0f;

    pointSphereSPHMaterial.setDiffuse(QVector4D(0.82f, 0.45f, 1.0f, 1.0f));
    pointSphereSPHMaterial.setSpecular(QVector4D(0.5f, 0.5f, 0.5f, 1.0f));
    pointSphereSPHMaterial.shininess = 150.0f;

    pointSpherePDMaterial.setDiffuse(QVector4D(0.42f, 1.0f, 0.5f, 1.0f));
    pointSpherePDMaterial.setSpecular(QVector4D(0.5f, 0.5f, 0.5f, 1.0f));
    pointSpherePDMaterial.shininess = 150.0f;

    /////////////////////////////////////////////////////////////////
    // setup binding points for block uniform
    for(int i = 0; i < NUM_BINDING_POINTS; ++i)
    {
        UBOBindingIndex[i] = i + 1;
    }

    /////////////////////////////////////////////////////////////////
    // setup data for block uniform
    glGenBuffers(1, &UBOMatrices);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferData(GL_UNIFORM_BUFFER, 4 * SIZE_OF_MAT4, NULL,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOLight);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOLight);
    glBufferData(GL_UNIFORM_BUFFER, light.getStructSize(),
                 &light, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOSurfaceMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOSurfaceMaterial);
    glBufferData(GL_UNIFORM_BUFFER, surfaceMaterial.getStructSize(),
                 &surfaceMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOPointSphereSPHMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOPointSphereSPHMaterial);
    glBufferData(GL_UNIFORM_BUFFER, pointSphereSPHMaterial.getStructSize(),
                 &pointSphereSPHMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOPointSpherePDMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOPointSpherePDMaterial);
    glBufferData(GL_UNIFORM_BUFFER, pointSpherePDMaterial.getStructSize(),
                 &pointSpherePDMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOGroundMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOGroundMaterial);
    glBufferData(GL_UNIFORM_BUFFER, groundMaterial.getStructSize(),
                 &groundMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    glGenBuffers(1, &UBOBoxMaterial);
    glBindBuffer(GL_UNIFORM_BUFFER, UBOBoxMaterial);
    glBufferData(GL_UNIFORM_BUFFER, boxMaterial.getStructSize(),
                 &boxMaterial, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

//------------------------------------------------------------------------------------------
void Renderer::initTexture()
{
    ////////////////////////////////////////////////////////////////////////////////
    // environment texture
    QMap<EnvironmentTexture, QString> envTexture2StrMap;
    envTexture2StrMap[SKY1] = "sky2";

    TRUE_OR_DIE(envTexture2StrMap.size() == (NUM_ENVIRONMENT_TEXTURES - 1),
                "Ohh, you forget to initialize some environment texture...");

    for(int i = 1; i < NUM_ENVIRONMENT_TEXTURES; ++i)
    {
        EnvironmentTexture tex = static_cast<EnvironmentTexture>(i);

        QString posXFile = QString(":/textures/%1/posx.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(posXFile), "Cannot load texture from file.");

        QString negXFile = QString(":/textures/%1/negx.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(negXFile), "Cannot load texture from file.");

        QString posYFile = QString(":/textures/%1/posy.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(posYFile), "Cannot load texture from file.");

        QString negYFile = QString(":/textures/%1/negy.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(negYFile), "Cannot load texture from file.");

        QString posZFile = QString(":/textures/%1/posz.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(posZFile), "Cannot load texture from file.");

        QString negZFile = QString(":/textures/%1/negz.jpg").arg(envTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(negZFile), "Cannot load texture from file.");

        QImage posXTex = QImage(posXFile).convertToFormat(QImage::Format_RGBA8888);
        QImage negXTex = QImage(negXFile).convertToFormat(QImage::Format_RGBA8888);
        QImage posYTex = QImage(posYFile).convertToFormat(QImage::Format_RGBA8888);
        QImage negYTex = QImage(negYFile).convertToFormat(QImage::Format_RGBA8888);
        QImage posZTex = QImage(posZFile).convertToFormat(QImage::Format_RGBA8888);
        QImage negZTex = QImage(negZFile).convertToFormat(QImage::Format_RGBA8888);


        cubeMapEnvTexture[i] = new QOpenGLTexture(QOpenGLTexture::TargetCubeMap);
        cubeMapEnvTexture[i]->create();
        cubeMapEnvTexture[i]->setSize(posXTex.width(), posXTex.height());
        cubeMapEnvTexture[i]->setFormat(QOpenGLTexture::RGBA8_UNorm);
        cubeMapEnvTexture[i]->allocateStorage();

        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapPositiveX,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posXTex.constBits());
        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapNegativeX,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negXTex.constBits());
        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapPositiveY,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posYTex.constBits());
        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapNegativeY,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negYTex.constBits());
        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapPositiveZ,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, posZTex.constBits());
        cubeMapEnvTexture[i]->setData(0, 0, QOpenGLTexture::CubeMapNegativeZ,
                                      QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, negZTex.constBits());

        cubeMapEnvTexture[i]->setWrapMode(QOpenGLTexture::DirectionS,
                                          QOpenGLTexture::ClampToEdge);
        cubeMapEnvTexture[i]->setWrapMode(QOpenGLTexture::DirectionT,
                                          QOpenGLTexture::ClampToEdge);
        cubeMapEnvTexture[i]->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
        cubeMapEnvTexture[i]->setMagnificationFilter(QOpenGLTexture::LinearMipMapLinear);
    }

    if(QOpenGLContext::currentContext()->hasExtension("GL_ARB_seamless_cube_map"))
    {
        qDebug() << "GL_ARB_seamless_cube_map: enabled";
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    }
    else
    {
//        qDebug() << "GL_ARB_seamless_cube_map: disabled";
        glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    }


    ////////////////////////////////////////////////////////////////////////////////
    // floor texture
    QMap<FloorTexture, QString> floorTexture2StrMap;
    floorTexture2StrMap[CHECKERBOARD1] = "checkerboard1.jpg";
    floorTexture2StrMap[CHECKERBOARD2] = "checkerboard2.jpg";
    floorTexture2StrMap[STONE1] = "stone1.jpg";
    floorTexture2StrMap[STONE2] = "stone2.jpg";
    floorTexture2StrMap[WOOD1] = "wood1.jpg";
    floorTexture2StrMap[WOOD2] = "wood2.jpg";

    TRUE_OR_DIE(floorTexture2StrMap.size() == (NUM_FLOOR_TEXTURES - 1),
                "Ohh, you forget to initialize some floor texture...");

    for(int i = 0; i < NUM_FLOOR_TEXTURES; ++i)
    {
        FloorTexture tex = static_cast<FloorTexture>(i);

        QString texFile = QString(":/textures/%1").arg(floorTexture2StrMap[tex]);
        TRUE_OR_DIE(QFile::exists(texFile), "Cannot load texture from file.");
        floorTextures[tex] = new QOpenGLTexture(QImage(texFile).mirrored());
        floorTextures[tex]->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
        floorTextures[tex]->setMagnificationFilter(QOpenGLTexture::LinearMipMapLinear);
        floorTextures[tex]->setWrapMode(QOpenGLTexture::Repeat);
    }
}

//------------------------------------------------------------------------------------------
void Renderer::initSceneMemory()
{
    initBackgroundMemory();
    initGroundMemory();
    initBoxMemory();
    initLightObjectMemory();
    //initParticlesMemory();
}

//------------------------------------------------------------------------------------------
void Renderer::initBackgroundMemory()
{
    if(!cubeObject)
    {
        cubeObject = new UnitCube;
    }

    if(vboBackground.isCreated())
    {
        vboBackground.destroy();
    }

    if(iboBackground.isCreated())
    {
        iboBackground.destroy();
    }

    vboBackground.create();
    vboBackground.bind();

    ////////////////////////////////////////////////////////////////////////////////
    // init memory for background cube
    vboBackground.create();
    vboBackground.bind();

    vboBackground.allocate(cubeObject->getVertexOffset());
    vboBackground.write(0, cubeObject->getVertices(), cubeObject->getVertexOffset());
    vboBackground.release();
// indices
    iboBackground.create();
    iboBackground.bind();
    iboBackground.allocate(cubeObject->getIndices(), cubeObject->getIndexOffset());
    iboBackground.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initGroundMemory()
{
    if(!planeObject)
    {
        planeObject = new UnitPlane;
    }

    if(vboGround.isCreated())
    {
        vboGround.destroy();
    }

    if(iboGround.isCreated())
    {
        iboGround.destroy();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // init memory for billboard object
    vboGround.create();
    vboGround.bind();
    vboGround.allocate(2 * planeObject->getVertexOffset() +
                       planeObject->getTexCoordOffset());
    vboGround.write(0, planeObject->getVertices(), planeObject->getVertexOffset());
    vboGround.write(planeObject->getVertexOffset(), planeObject->getNormals(),
                    planeObject->getVertexOffset());
    vboGround.write(2 * planeObject->getVertexOffset(),
                    planeObject->getTexureCoordinates(GROUND_PLANE_SIZE),
                    planeObject->getTexCoordOffset());
    vboGround.release();
    // indices
    iboGround.create();
    iboGround.bind();
    iboGround.allocate(planeObject->getIndices(), planeObject->getIndexOffset());
    iboGround.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initBoxMemory()
{
    if(!cubeObject)
    {
        cubeObject = new UnitCube;
    }

    if(vboBox.isCreated())
    {
        vboBox.destroy();
    }

    if(iboBox.isCreated())
    {
        iboBox.destroy();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // init memory for cube
    vboBox.create();
    vboBox.bind();
    vboBox.allocate(2 * cubeObject->getVertexOffset() + cubeObject->getTexCoordOffset());
    vboBox.write(0, cubeObject->getVertices(), cubeObject->getVertexOffset());
    vboBox.write(cubeObject->getVertexOffset(), cubeObject->getNormals(),
                 cubeObject->getVertexOffset());
    vboBox.write(2 * cubeObject->getVertexOffset(), cubeObject->getTexureCoordinates(1.0f),
                 cubeObject->getTexCoordOffset());
    vboBox.release();
    // indices
    iboBox.create();
    iboBox.bind();
    iboBox.allocate(cubeObject->getLineIndices(), cubeObject->getLineIndexOffset());
    iboBox.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initLightObjectMemory()
{
    if(vboLight.isCreated())
    {
        vboLight.destroy();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // init memory for cube
    vboLight.create();
    vboLight.bind();
    vboLight.allocate(3 * sizeof(GLfloat));
    QVector3D lightPos(DEFAULT_LIGHT_DIRECTION);
    vboLight.write(0, &lightPos, 3 * sizeof(GLfloat));
    vboLight.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initParticlesRandomColors()
{
    ////////////////////////////////////////////////////////////////////////////////
    // init memory for random color
    srand(15646);

    if(particleRandomColors)
    {
        delete[] particleRandomColors;
    }

    particleRandomColors = new GLREAL[num_particles_ * 3];

    for(int i = 0; i < num_particles_; ++i)
    {
        (particleRandomColors)[i * 3 + 0] = (GLREAL) (rand() / (GLREAL) RAND_MAX);
        (particleRandomColors)[i * 3 + 1] = (GLREAL) (rand() / (GLREAL) RAND_MAX);
        (particleRandomColors)[i * 3 + 2] = (GLREAL) (rand() / (GLREAL) RAND_MAX);
    }

}

//------------------------------------------------------------------------------------------
void Renderer::initParticlesRampColors()
{
    ////////////////////////////////////////////////////////////////////////////////
    // init memory for ramp color
    if(particleRampColors)
    {
        delete[] particleRampColors;
    }

    particleRampColors = new GLREAL[num_particles_ * 3];

    float t;

    for (int i = 0; i < num_pd_particles_; ++i)
    {
        t = (float)i / (float) num_pd_particles_;
        colorRamp(t, &(particleRampColors)[i * 3]);
    }

    for (int i = 0; i < num_sph_particles_; ++i)
    {
        t = (float)i / (float) num_sph_particles_;
        colorRamp(t, &(particleRampColors)[(i + num_pd_particles_) * 3]);
    }

}

//------------------------------------------------------------------------------------------
void Renderer::initVertexArrayObjects()
{
    initBackgroundVAO();
    initGroundVAO();
    initBoxVAO();
    initLightVAO();

//        initParticlesVAO(PROGRAM_POINT_SPHERE_VIEW);
}

//------------------------------------------------------------------------------------------
void Renderer::initBackgroundVAO()
{
    if(vaoBackground.isCreated())
    {
        vaoBackground.destroy();
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_BACKGROUND];

    vaoBackground.create();
    vaoBackground.bind();

    vboBackground.bind();
    program->enableAttributeArray(attrVertex[PROGRAM_RENDER_BACKGROUND]);
    program->setAttributeBuffer(attrVertex[PROGRAM_RENDER_BACKGROUND], GL_FLOAT, 0, 3);

    iboBackground.bind();

    // release vao before vbo and ibo
    vaoBackground.release();
    vboBackground.release();
    iboBackground.release();

}

//------------------------------------------------------------------------------------------
void Renderer::initGroundVAO()
{
    if(vaoGround.isCreated())
    {
        vaoGround.destroy();
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_GROUND];

    vaoGround.create();
    vaoGround.bind();

    vboGround.bind();
    program->enableAttributeArray(attrVertex[PROGRAM_RENDER_GROUND]);
    program->setAttributeBuffer(attrVertex[PROGRAM_RENDER_GROUND], GL_FLOAT, 0, 3);

    program->enableAttributeArray(attrNormal[PROGRAM_RENDER_GROUND]);
    program->setAttributeBuffer(attrNormal[PROGRAM_RENDER_GROUND], GL_FLOAT,
                                planeObject->getVertexOffset(), 3);

    program->enableAttributeArray(attrTexCoord[PROGRAM_RENDER_GROUND]);
    program->setAttributeBuffer(attrTexCoord[PROGRAM_RENDER_GROUND], GL_FLOAT,
                                2 * planeObject->getVertexOffset(), 2);

    iboGround.bind();

    // release vao before vbo and ibo
    vaoGround.release();
    vboGround.release();
    iboGround.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initBoxVAO()
{
    if(vaoBox.isCreated())
    {
        vaoBox.destroy();
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_BOX];

    vaoBox.create();
    vaoBox.bind();

    vboBox.bind();
    program->enableAttributeArray(attrVertex[PROGRAM_RENDER_BOX]);
    program->setAttributeBuffer(attrVertex[PROGRAM_RENDER_BOX], GL_FLOAT, 0, 3);

    program->enableAttributeArray(attrNormal[PROGRAM_RENDER_BOX]);
    program->setAttributeBuffer(attrNormal[PROGRAM_RENDER_BOX], GL_FLOAT,
                                cubeObject->getVertexOffset(), 3);

    program->enableAttributeArray(attrTexCoord[PROGRAM_RENDER_BOX]);
    program->setAttributeBuffer(attrTexCoord[PROGRAM_RENDER_BOX], GL_FLOAT,
                                2 * cubeObject->getVertexOffset(), 2);

    iboBox.bind();

    // release vao before vbo and ibo
    vaoBox.release();
    vboBox.release();
    iboBox.release();

}

//------------------------------------------------------------------------------------------
void Renderer::initLightVAO()
{
    if(vaoLight.isCreated())
    {
        vaoLight.destroy();
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_LIGHT];

    vaoLight.create();
    vaoLight.bind();

    vboLight.bind();
    program->enableAttributeArray(attrVertex[PROGRAM_RENDER_LIGHT]);
    program->setAttributeBuffer(attrVertex[PROGRAM_RENDER_LIGHT], GL_FLOAT, 0, 3);

    // release vao before vbo and ibo
    vaoLight.release();
    vboLight.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initParticlesVAO(GLSLPrograms program_type)
{
    if(vaoParticles.isCreated())
    {
        vaoParticles.destroy();
    }

    QOpenGLShaderProgram* program = glslPrograms[program_type];

    vaoParticles.create();
    vaoParticles.bind();

    vboParticles.bind();

    program->enableAttributeArray(attrVertex[program_type]);
    program->setAttributeBuffer(attrVertex[program_type], GL_REAL, 0, 4);

    program->enableAttributeArray(attrColor[program_type]);
    program->setAttributeBuffer(attrColor[program_type], GL_REAL,
                                num_particles_ * sizeof(GLREAL) * 4, 3);

    program->enableAttributeArray(attrActivity[program_type]);
//    program->setAttributeBuffer(attrActivity[program_type], GL_INT,
//                                num_particles_ * sizeof(GLREAL) * (4 + 3), 1);

    // use this command to avoid converting int to float
    glVertexAttribIPointer(attrActivity[program_type],
                           1, GL_INT, 0,
                           (GLvoid*) ( num_particles_ * sizeof(GLREAL) * (4 + 3)));

    // release vao before vbo and ibo
    vaoParticles.release();
    vboParticles.release();
}

//------------------------------------------------------------------------------------------
void Renderer::initFrameBufferObject()
{
//    if(depthTexture)
//    {
//        depthTexture->destroy();
//        delete depthTexture;
//    }

    depthTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
    depthTexture->create();
    depthTexture->setSize(DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
    depthTexture->setFormat(QOpenGLTexture::D32);
    depthTexture->allocateStorage();
    depthTexture->setMinificationFilter(QOpenGLTexture::Linear);
    depthTexture->setMagnificationFilter(QOpenGLTexture::Linear);
    depthTexture->setWrapMode(QOpenGLTexture::DirectionS,
                              QOpenGLTexture::ClampToEdge);
    depthTexture->setWrapMode(QOpenGLTexture::DirectionT,
                              QOpenGLTexture::ClampToEdge);

    depthTexture->bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

    // frame buffer
    FBODepth = new QOpenGLFramebufferObject(DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
    FBODepth->bind();
//    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
//                         dTex, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                         depthTexture->textureId(), 0);
    TRUE_OR_DIE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE,
                "Framebuffer is imcomplete!");
    FBODepth->release();

}

//------------------------------------------------------------------------------------------
void Renderer::updateCamera()
{
    zoomCamera();

    /////////////////////////////////////////////////////////////////
    // flush camera data to uniform buffer
    viewMatrix.setToIdentity();
    viewMatrix.lookAt(cameraPosition_, cameraFocus_, cameraUpDirection_);


    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 2 * SIZE_OF_MAT4, SIZE_OF_MAT4,
                    viewMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

//------------------------------------------------------------------------------------------
void Renderer::translateCamera()
{
    translation_ *= MOVING_INERTIA;

    if(translation_.lengthSquared() < 1e-4)
    {
        return;
    }

    QVector3D eyeVector = cameraFocus_ - cameraPosition_;
    float scale = sqrt(eyeVector.length()) * 0.01f;

    QVector3D u = cameraUpDirection_;
    QVector3D v = QVector3D::crossProduct(eyeVector, u);
    u = QVector3D::crossProduct(v, eyeVector);
    u.normalize();
    v.normalize();

    cameraPosition_ -= scale * (translation_.x() * v + translation_.y() * u);
    cameraFocus_ -= scale * (translation_.x() * v + translation_.y() * u);

}

//------------------------------------------------------------------------------------------
void Renderer::rotateCamera()
{
    rotation_ *= MOVING_INERTIA;

    if(rotation_.lengthSquared() < 1e-4)
    {
        return;
    }

    QVector3D nEyeVector = cameraPosition_ - cameraFocus_ ;
    QVector3D u = cameraUpDirection_;
    QVector3D v = QVector3D::crossProduct(-nEyeVector, u);

    u = QVector3D::crossProduct(v, -nEyeVector);
    u.normalize();
    v.normalize();

    float scale = sqrt(nEyeVector.length()) * 0.02f;
    QQuaternion qRotation = QQuaternion::fromAxisAndAngle(v, rotation_.y() * scale) *
                            QQuaternion::fromAxisAndAngle(u, rotation_.x() * scale) *
                            QQuaternion::fromAxisAndAngle(nEyeVector, rotation_.z() * scale);
    nEyeVector = qRotation.rotatedVector(nEyeVector);

    QQuaternion qCamRotation = QQuaternion::fromAxisAndAngle(v, rotation_.y() * scale) *
                               QQuaternion::fromAxisAndAngle(nEyeVector, rotation_.z() * scale);

    cameraPosition_ = cameraFocus_ + nEyeVector;
    cameraUpDirection_ = qCamRotation.rotatedVector(cameraUpDirection_);
}

//------------------------------------------------------------------------------------------
void Renderer::zoomCamera()
{
    zooming_ *= MOVING_INERTIA;

    if(fabs(zooming_) < 1e-4)
    {
        return;
    }

    QVector3D nEyeVector = cameraPosition_ - cameraFocus_ ;
    float len = nEyeVector.length();
    nEyeVector.normalize();

    len += sqrt(len) * zooming_ * 0.1f;

    if(len < 0.1f)
    {
        len = 0.1f;
    }

    cameraPosition_ = len * nEyeVector + cameraFocus_;

}

//------------------------------------------------------------------------------------------
void Renderer::translateLight()
{
    translation_ *= MOVING_INERTIA;

    if(translation_.lengthSquared() < 1e-4)
    {
        return;
    }

    QVector3D eyeVector = cameraFocus_ - cameraPosition_;
    float scale = sqrt(eyeVector.length()) * 0.05f;

    QVector3D u(0.0f, 1.0f, 0.0f);
    QVector3D v = QVector3D::crossProduct(eyeVector, u);
    u = QVector3D::crossProduct(v, eyeVector);
    u.normalize();
    v.normalize();

    QVector3D objectTrans = scale * (translation_.x() * v + translation_.y() * u);
    QMatrix4x4 translationMatrix;
    translationMatrix.setToIdentity();
    translationMatrix.translate(objectTrans);

    light.position = translationMatrix * light.position;
    glBindBuffer(GL_UNIFORM_BUFFER, UBOLight);
    glBufferData(GL_UNIFORM_BUFFER, light.getStructSize(),
                 &light, GL_STREAM_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}
//------------------------------------------------------------------------------------------
void Renderer::renderScene()
{

    glViewport(0, 0,  width() * retinaScale, height() * retinaScale);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderBackground();
    renderGround();
    renderLight();
    renderBox();

    if(!isParticlesReady)
    {
        return;
    }

    switch(currentParticleViewMode)
    {
    case POINTS_VIEW:
        renderParticlesAsPointSphere(true);
        break;

    case SPHERES_VIEW:
        renderParticlesAsPointSphere(false);
        break;

    case OPAQUE_SURFACE_VIEW:
        renderParticlesAsSurface(0.0);
        break;

    case TRANSPARENT_SURFACE_VIEW:
        renderParticlesAsSurface(0.8);
        break;

    }

    if(enabledImageOutput && !pausedImageOutput)
    {
        exportScreenToImage();
    }
}

//------------------------------------------------------------------------------------------
void Renderer::renderBackground()
{
    if(currentEnvironmentTexture == NO_ENVIRONMENT_TEXTURE)
    {
        return;
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_BACKGROUND];
    program->bind();

    /////////////////////////////////////////////////////////////////
    // flush the model matrices
    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, SIZE_OF_MAT4,
                    backgroundCubeModelMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
    /////////////////////////////////////////////////////////////////
    // set the uniform
    program->setUniformValue(uniCameraPosition[PROGRAM_RENDER_BACKGROUND], cameraPosition_);
    program->setUniformValue(uniObjTexture[PROGRAM_RENDER_BACKGROUND], 0);

    glUniformBlockBinding(program->programId(), uniMatrices[PROGRAM_RENDER_BACKGROUND],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);
    /////////////////////////////////////////////////////////////////
    // render the background
    vaoBackground.bind();
    cubeMapEnvTexture[currentEnvironmentTexture]->bind(0);
    glDrawElements(GL_TRIANGLES, cubeObject->getNumIndices(), GL_UNSIGNED_SHORT, 0);
    cubeMapEnvTexture[currentEnvironmentTexture]->release();
    vaoBackground.release();
    program->release();

}

//------------------------------------------------------------------------------------------
void Renderer::renderGround()
{
    if(currentFloorTexture == NO_FLOOR)
    {
        return;
    }

    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, SIZE_OF_MAT4,
                    groundModelMatrix.constData());
    glBufferSubData(GL_UNIFORM_BUFFER, SIZE_OF_MAT4, SIZE_OF_MAT4,
                    groundNormalMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_GROUND];
    program->bind();
    /////////////////////////////////////////////////////////////////
    // set the uniform
    program->setUniformValue(uniHasObjTexture[PROGRAM_RENDER_GROUND], GL_TRUE);
    program->setUniformValue(uniObjTexture[PROGRAM_RENDER_GROUND], 0);
    program->setUniformValue(uniCameraPosition[PROGRAM_RENDER_GROUND], cameraPosition_);

    glUniformBlockBinding(program->programId(), uniMatrices[PROGRAM_RENDER_GROUND],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);
    glUniformBlockBinding(program->programId(), uniLight[PROGRAM_RENDER_GROUND],
                          UBOBindingIndex[BINDING_LIGHT]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_LIGHT],
                     UBOLight);
    glUniformBlockBinding(program->programId(), uniMaterial[PROGRAM_RENDER_GROUND],
                          UBOBindingIndex[BINDING_GROUND_MATERIAL]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_GROUND_MATERIAL],
                     UBOGroundMaterial);

    /////////////////////////////////////////////////////////////////
    // render the floor
    vaoGround.bind();
    floorTextures[currentFloorTexture]->bind(0);

    if(bTextureAnisotropicFiltering)
    {
        GLfloat fLargest;
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &fLargest);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, fLargest);
    }
    else
    {
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 1.0f);
    }


    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
    floorTextures[currentFloorTexture]->release();
    vaoGround.release();
}

//------------------------------------------------------------------------------------------
void Renderer::renderLight()
{
    if(!vaoLight.isCreated())
    {
        qDebug() << "vaoLight is not created!";
        return;
    }

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_LIGHT];
    program->bind();

    /////////////////////////////////////////////////////////////////
    // set the uniform
    glUniformBlockBinding(program->programId(), uniMatrices[PROGRAM_RENDER_LIGHT],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);

    glUniformBlockBinding(program->programId(), uniLight[PROGRAM_RENDER_LIGHT],
                          UBOBindingIndex[BINDING_LIGHT]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_LIGHT],
                     UBOLight);

    program->setUniformValue("pointDistance",
                             (cameraPosition_ - cameraFocus_).length());
    vaoLight.bind();
    glEnable (GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable (GL_DEPTH_TEST);
    glDrawArrays(GL_POINTS, 0, 1);
    glDisable(GL_POINT_SPRITE);

    vaoLight.release();
    program->release();
}

//------------------------------------------------------------------------------------------
void Renderer::renderBox()
{
    if(!vaoBox.isCreated())
    {
        qDebug() << "vaoBox is not created!";
        return;
    }

    /////////////////////////////////////////////////////////////////
    // flush the model and normal matrices
    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, SIZE_OF_MAT4,
                    boxModelMatrix.constData());
    glBufferSubData(GL_UNIFORM_BUFFER, SIZE_OF_MAT4, SIZE_OF_MAT4,
                    boxNormalMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_BOX];
    program->bind();
    /////////////////////////////////////////////////////////////////
    // set the uniform
    program->setUniformValue(uniHasObjTexture[PROGRAM_RENDER_BOX], GL_FALSE);
    program->setUniformValue("lineView", GL_TRUE);

    glUniformBlockBinding(program->programId(), uniMatrices[PROGRAM_RENDER_BOX],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);
    glUniformBlockBinding(program->programId(), uniLight[PROGRAM_RENDER_BOX],
                          UBOBindingIndex[BINDING_LIGHT]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_LIGHT],
                     UBOLight);

    glUniformBlockBinding(program->programId(), uniMaterial[PROGRAM_RENDER_BOX],
                          UBOBindingIndex[BINDING_BOX_MATERIAL]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_BOX_MATERIAL],
                     UBOBoxMaterial);

    /////////////////////////////////////////////////////////////////
    // render the cube
    vaoBox.bind();
//    glDrawElements(GL_LINES, cubeObject->getNumIndices(), GL_UNSIGNED_SHORT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//    glDrawElements(GL_TRIANGLES, cubeObject->getNumIndices(), GL_UNSIGNED_SHORT, 0);
    glDrawElements(GL_LINES, cubeObject->getNumLineIndices(), GL_UNSIGNED_SHORT, 0);
//        glDrawElements(GL_LINES, 8*2, GL_UNSIGNED_SHORT, 0);
    vaoBox.release();
}

//------------------------------------------------------------------------------------------
void Renderer::renderParticlesAsPointSphere(bool bPointView)
{
    if(!vaoParticles.isCreated())
    {
        qDebug() << "vaoParticles is not created!";
        return;
    }

    /////////////////////////////////////////////////////////////////
    // flush the model and normal matrices
    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, SIZE_OF_MAT4,
                    particlesModelMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_POINT_SPHERE_VIEW];
    program->bind();

    /////////////////////////////////////////////////////////////////
    // set the uniform
    glUniformBlockBinding(program->programId(), uniMatrices[PROGRAM_POINT_SPHERE_VIEW],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);

    glUniformBlockBinding(program->programId(), uniLight[PROGRAM_POINT_SPHERE_VIEW],
                          UBOBindingIndex[BINDING_LIGHT]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_LIGHT],
                     UBOLight);

//    qDebug() << bHideInvisibleParticles;
    program->setUniformValue("pointView", bPointView);
    program->setUniformValue("hideInvisibleParticles", bHideInvisibleParticles);
    program->setUniformValue("cameraPosition", cameraPosition_);
    program->setUniformValue("pointScale",
                             height() / tanf(45 * 0.5f * (float) M_PI / 180.0f));

    if(currentParticleColorMode == COLOR_RANDOM ||
       currentParticleColorMode == COLOR_RAMP ||
       currentParticleColorMode == COLOR_DENSITY ||
       currentParticleColorMode == COLOR_STIFFNESS ||
       currentParticleColorMode == COLOR_ACTIVITY)
    {
        program->setUniformValue("hasVertexColor", GL_TRUE);
    }
    else
    {
        program->setUniformValue("hasVertexColor", GL_FALSE);
    }

    vaoParticles.bind();
    glEnable (GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable (GL_DEPTH_TEST);


    glUniformBlockBinding(program->programId(), uniMaterial[PROGRAM_POINT_SPHERE_VIEW],
                          UBOBindingIndex[BINDING_POINT_SPHERE_PD_MATERIAL]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_POINT_SPHERE_PD_MATERIAL],
                     UBOPointSpherePDMaterial);
    program->setUniformValue("pointRadius", pd_particle_radius_);
    glDrawArrays(GL_POINTS, 0, num_pd_particles_);

    glUniformBlockBinding(program->programId(), uniMaterial[PROGRAM_POINT_SPHERE_VIEW],
                          UBOBindingIndex[BINDING_POINT_SPHERE_SPH_MATERIAL]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_POINT_SPHERE_SPH_MATERIAL],
                     UBOPointSphereSPHMaterial);
    program->setUniformValue("pointRadius", sph_particle_radius_);
    glDrawArrays(GL_POINTS, num_pd_particles_, num_sph_particles_);

    glDisable(GL_POINT_SPRITE);
    vaoParticles.release();
    program->release();
}

//------------------------------------------------------------------------------------------
void Renderer::renderParticlesAsSurface(float transparency)
{
    if(!vaoParticles.isCreated())
    {
        qDebug() << "vaoParticles is not created!";
        return;
    }

    renderParticlesToDepthBuffer();

    // render surface
    makeCurrent();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);



}

//------------------------------------------------------------------------------------------
void Renderer::renderParticlesToDepthBuffer()
{
//    FBODepth->bind();
//    glViewport(0, 0, DEPTH_TEXTURE_SIZE, DEPTH_TEXTURE_SIZE);
//    glDrawBuffer(GL_NONE);

    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_UNIFORM_BUFFER, UBOMatrices);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, SIZE_OF_MAT4,
                    particlesModelMatrix.constData());
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    QOpenGLShaderProgram* program = glslPrograms[PROGRAM_RENDER_DEPTH_BUFFER];
    program->bind();
    glUniformBlockBinding(program->programId(),
                          uniMatrices[PROGRAM_RENDER_DEPTH_BUFFER],
                          UBOBindingIndex[BINDING_MATRICES]);
    glBindBufferBase(GL_UNIFORM_BUFFER, UBOBindingIndex[BINDING_MATRICES],
                     UBOMatrices);

//    program->setUniformValue("cameraPosition", cameraPosition);
    program->setUniformValue("pointScale",
                             height() / tanf(45 * 0.5f * (float) M_PI / 180.0f));


    vaoParticles.bind();
    glEnable (GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable (GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    program->setUniformValue("pointRadius", pd_particle_radius_);
    glDrawArrays(GL_POINTS, 0, num_pd_particles_);

    program->setUniformValue("pointRadius", sph_particle_radius_);
    glDrawArrays(GL_POINTS, num_pd_particles_, num_sph_particles_);

    glDisable(GL_POINT_SPRITE);
    vaoParticles.release();


    program->release();

    FBODepth->release();
}

//------------------------------------------------------------------------------------------
void Renderer::exportScreenToImage()
{
    glReadPixels(0, 0, width(), height(), GL_RGBA, GL_UNSIGNED_BYTE, outputImage->bits());
    outputImage->mirrored().save(QString(imageOutputPath + "/frame.%1.png").arg(
                                     current_frame_));
    qDebug() << "Saved image: " << current_frame_;
}

//------------------------------------------------------------------------------------------

