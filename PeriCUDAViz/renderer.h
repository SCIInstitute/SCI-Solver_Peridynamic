//------------------------------------------------------------------------------------------
//
//
// Created on: 2/20/2015
//     Author: Nghia Truong
//
//------------------------------------------------------------------------------------------

#ifndef RENDERER_H
#define RENDERER_H

#include <QtGui>
#include <QtWidgets>
#include <QOpenGLFunctions_4_1_Core>

#include "parameters.h"
#include "unitcube.h"
#include "unitplane.h"


//------------------------------------------------------------------------------------------
#define PRINT_LINE \
{ \
    qDebug()<< "Line:" << __LINE__ << ", file:" << __FILE__; \
}

#define PRINT_ERROR(_errStr) \
{ \
    qDebug()<< "Error occured at line:" << __LINE__ << ", file:" << __FILE__; \
    qDebug()<< "Error message:" << _errStr; \
}

#define PRINT_AND_DIE(_errStr) \
{ \
    qDebug()<< "Error occured at line:" << __LINE__ << ", file:" << __FILE__; \
    qDebug()<< "Error message:" << _errStr; \
    exit(EXIT_FAILURE); \
}

#define TRUE_OR_DIE(_condition, _errStr) \
{ \
    if(!(_condition)) \
    { \
        qDebug()<< "Fatal error occured at line:" << __LINE__ << ", file:" << __FILE__; \
        qDebug()<< "Error message:" << _errStr; \
        exit(EXIT_FAILURE); \
    } \
}

#define SIZE_OF_MAT4 (4 * 4 *sizeof(GLfloat))
#define SIZE_OF_VEC4 (4 * sizeof(GLfloat))
//------------------------------------------------------------------------------------------
#define NUM_SURFACE_SMOOTHING_ITERATION 5
#define MOVING_INERTIA 0.8f
#define GROUND_PLANE_SIZE 10.0F
#define DEPTH_TEXTURE_SIZE 1024
#define DEFAULT_CAMERA_POSITION QVector3D(-3.5f, 1.0f, 2.0f)
#define DEFAULT_CAMERA_FOCUS QVector3D(1.0f,  1.0f, 2.0f)
#define DEFAULT_LIGHT_DIRECTION QVector4D(10.0f, 12.0f, 6.0f, 1.0f)
#define DEFAULT_BOX_POSITION QVector3D(1.0f, 1.0f, 1.0f)
#define DEFAULT_GROUND_POSITION QVector3D(1.0f, 0.0f, 1.0f)


struct Light
{
    Light():
        position(10.0f, 10.0f, 10.0f, 1.0f),
        color(1.0f, 1.0f, 1.0f, 1.0f),
        intensity(1.0f) {}

    int getStructSize()
    {
        return (2 * 4 + 1) * sizeof(GLfloat);
    }

    QVector4D position;
    QVector4D color;
    GLfloat intensity;
};

struct Material
{
    Material():
        diffuseColor(-10.0f, 1.0f, 0.0f, 1.0f),
        specularColor(1.0f, 1.0f, 1.0f, 1.0f),
        reflection(0.0f),
        shininess(10.0f) {}

    int getStructSize()
    {
        return (2 * 4 + 2) * sizeof(GLfloat);
    }

    void setDiffuse(QVector4D _diffuse)
    {
        diffuseColor = _diffuse;
    }

    void setSpecular(QVector4D _specular)
    {
        specularColor = _specular;
    }

    void setReflection(float _reflection)
    {
        reflection = _reflection;
    }

    QVector4D diffuseColor;
    QVector4D specularColor;
    GLfloat reflection;
    GLfloat shininess;
};

enum ParticleViewMode
{
    POINTS_VIEW = 0,
    SPHERES_VIEW,
    OPAQUE_SURFACE_VIEW,
    TRANSPARENT_SURFACE_VIEW,
    NUM_VIEWING_MODE
};

enum GLSLPrograms
{
    PROGRAM_POINT_SPHERE_VIEW = 0,
    PROGRAM_SURFACE_VIEW,
    PROGRAM_RENDER_BACKGROUND,
    PROGRAM_RENDER_GROUND,
    PROGRAM_RENDER_BOX,
    PROGRAM_RENDER_LIGHT,
    PROGRAM_RENDER_DEPTH_BUFFER,
    NUM_GLSL_PROGRAMS
};

enum ParticleColorMode
{
    COLOR_RANDOM = 0,
    COLOR_RAMP,
    COLOR_PARTICLE_TYPE,
    COLOR_DENSITY,
    COLOR_STIFFNESS,
    COLOR_ACTIVITY,
    NUM_PARTICLE_COLOR_MODE
};

enum UBOBinding
{
    BINDING_MATRICES = 0,
    BINDING_LIGHT,
    BINDING_GROUND_MATERIAL,
    BINDING_BOX_MATERIAL,
    BINDING_SURFACE_MATERIAL,
    BINDING_POINT_SPHERE_SPH_MATERIAL,
    BINDING_POINT_SPHERE_PD_MATERIAL,
    NUM_BINDING_POINTS
};

enum MouseTransformationTarget
{
    TRANSFORM_CAMERA = 0,
    TRANSFORM_LIGHT,
    NUM_TRANSFORMATION_TARGET
};


enum FloorTexture
{
    NO_FLOOR = 0,
    CHECKERBOARD1,
    CHECKERBOARD2,
    STONE1,
    STONE2,
    WOOD1,
    WOOD2,
    NUM_FLOOR_TEXTURES
};

enum EnvironmentTexture
{
    NO_ENVIRONMENT_TEXTURE = 0,
    SKY1,
    NUM_ENVIRONMENT_TEXTURES
};

inline real_t lerp(real_t a, real_t b, real_t t)
{
    return a + t * (b - a);
}

inline void colorRamp(GLREAL t, GLREAL* r)
{
    const int ncolors = 7;
    real_t c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.0, 1.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
    };
    t = t * (ncolors - 1);
    int i = (int) t;
    real_t u = t - floor(t);
    r[0] = lerp(c[i][0], c[i + 1][0], u);
    r[1] = lerp(c[i][1], c[i + 1][1], u);
    r[2] = lerp(c[i][2], c[i + 1][2], u);
}

//------------------------------------------------------------------------------------------
class Renderer : public QOpenGLWidget, QOpenGLFunctions_4_1_Core
{
    Q_OBJECT
    friend class MainWindow;
public:
    explicit Renderer(QWidget* parent = 0, int viewportID = 0);

    enum SpecialKey
    {
        NO_KEY,
        SHIFT_KEY,
        CTRL_KEY
    };

    enum MouseButton
    {
        NO_BUTTON,
        LEFT_BUTTON,
        RIGHT_BUTTON
    };
    QSize sizeHint() const;
    QSize minimumSizeHint() const;

    void keyPressEvent(QKeyEvent*);
    void keyReleaseEvent(QKeyEvent*);
    void mousePressEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent*);
    void mouseReleaseEvent(QWheelEvent*);
    void wheelEvent(QWheelEvent*);

    void allocateMemory(int num_sph_particles, int num_pd_particles,
                        float sph_particle_radius, float pd_particle_radius);
    void setSPHParticleColor(float r, float g, float b);
    void setPDParticleColor(float r, float g, float b);
    void setSurfaceDiffuseColor(float r, float g, float b);
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

signals:
    void cameraChanged(QVector3D cameraPos, QVector3D cameraFocus_, QVector3D cameraUpDir);

public slots:
    void setCamera(QVector3D cameraPos, QVector3D cameraFocus, QVector3D cameraUpDir);

    void enableAnisotropicTextureFiltering(bool status);
    void enableClipYZPlane(bool status);
    void hideInvisibleParticles(bool status);


    void setParticleViewMode(int view_mode);
    void setParticleColorMode(int color_mode);
    void setParticlePositions(real_t* pd_positions, real_t* sph_positions,
                              int* pd_activities, int* sph_activities,
                              int current_frame);
    void setParticleDensitiesPositions(real_t* pd_positions, real_t* sph_positions,
                                       int* pd_activities, int* sph_activities,
                                       real_t* sph_densities, int current_frame);
    void setParticleStiffnesPositions(real_t* pd_positions, real_t* sph_positions,
                                      int* pd_activities, int* sph_activities,
                                      real_t* pd_stiffness, int current_frame);
    void setParticleActivitiesPositions(real_t* pd_positions, real_t* sph_positions,
                                        int* pd_activities, int* sph_activities, int current_frame);
    void setEnvironmentTexture(int texture);
    void setFloorTexture(int texture);
    void setMouseTransformationTarget(MouseTransformationTarget mouse_target);
    void setLightIntensity(int intensity);
    void resetCameraPosition();
    void resetLightPosition();
    void enableImageOutput(bool status);
    void pauseImageOutput(bool status);
    void setImageOutputPath(QString output_path);

private:
    void checkOpenGLVersion();
    void initScene();
    void initShaderPrograms();
    bool validateShaderPrograms(GLSLPrograms program);
    bool initSurfaceViewProgram();
    bool initPointSphereViewProgram();
    bool initRenderBackgroundProgram();
    bool initRenderGroundProgram();
    bool initRenderBoxProgram();
    bool initRenderLightProgram();
    bool initRenderDepthBufferProgram();

    void initSceneMatrices();
    void initSharedBlockUniform();
    void initTexture();
    void initSceneMemory();
    void initBackgroundMemory();
    void initGroundMemory();
    void initBoxMemory();
    void initLightObjectMemory();
    void initParticlesRandomColors();
    void initParticlesRampColors();

    void initVertexArrayObjects();
    void initBackgroundVAO();
    void initGroundVAO();
    void initBoxVAO();
    void initLightVAO();
    void initParticlesVAO(GLSLPrograms program_type);

    void initFrameBufferObject();

    void updateCamera();
    void translateCamera();
    void rotateCamera();
    void zoomCamera();

    void translateLight();

    void renderScene();
    void renderBackground();
    void renderGround();
    void renderLight();
    void renderBox();
    void renderParticlesAsPointSphere(bool bPointView);
    void renderParticlesAsSurface(float transparency);

    void renderParticlesToDepthBuffer();

    void exportScreenToImage();

    QOpenGLTexture* floorTextures[NUM_FLOOR_TEXTURES];
    QOpenGLTexture* cubeMapEnvTexture[NUM_ENVIRONMENT_TEXTURES];
    QMap<GLSLPrograms, QString> vertexShaderSourceMap;
    QMap<GLSLPrograms, QString> fragmentShaderSourceMap;
    QOpenGLShaderProgram* glslPrograms[NUM_GLSL_PROGRAMS];
    UnitPlane* planeObject;
    UnitCube* cubeObject;

    GLuint UBOBindingIndex[UBOBinding::NUM_BINDING_POINTS];
    GLuint UBOMatrices;
    GLuint UBOLight;
    GLuint UBOGroundMaterial;
    GLuint UBOBoxMaterial;
    GLuint UBOSurfaceMaterial;
    GLuint UBOPointSphereSPHMaterial;
    GLuint UBOPointSpherePDMaterial;

    GLint attrVertex[NUM_GLSL_PROGRAMS];
    GLint attrColor[NUM_GLSL_PROGRAMS];
    GLint attrActivity[NUM_GLSL_PROGRAMS];
    GLint attrNormal[NUM_GLSL_PROGRAMS];
    GLint attrTexCoord[NUM_GLSL_PROGRAMS];

    GLint uniMatrices[NUM_GLSL_PROGRAMS];
    GLint uniLight[NUM_GLSL_PROGRAMS];
    GLint uniCameraPosition[NUM_GLSL_PROGRAMS];
    GLint uniMaterial[NUM_GLSL_PROGRAMS];
    GLint uniObjTexture[NUM_GLSL_PROGRAMS];
    GLint uniHasObjTexture[NUM_GLSL_PROGRAMS];

    QOpenGLVertexArrayObject vaoBackground;
    QOpenGLVertexArrayObject vaoGround;
    QOpenGLVertexArrayObject vaoBox;
    QOpenGLVertexArrayObject vaoLight;
    QOpenGLBuffer vboBackground;
    QOpenGLBuffer vboGround;
    QOpenGLBuffer vboBox;
    QOpenGLBuffer vboLight;
    QOpenGLBuffer iboBackground;
    QOpenGLBuffer iboGround;
    QOpenGLBuffer iboBox;
    QOpenGLVertexArrayObject vaoParticles;
    QOpenGLBuffer vboParticles;

    QOpenGLFramebufferObject* FBODepth;
    QOpenGLTexture* depthTexture;

    Material groundMaterial;
    Material boxMaterial;
    Material surfaceMaterial;
    Material pointSphereSPHMaterial;
    Material pointSpherePDMaterial;
    Light light;

    QMatrix4x4 viewMatrix;
    QMatrix4x4 projectionMatrix;

    QMatrix4x4 particlesModelMatrix;
    QMatrix4x4 backgroundCubeModelMatrix;
    QMatrix4x4 boxModelMatrix;
    QMatrix4x4 boxNormalMatrix;
    QMatrix4x4 groundModelMatrix;
    QMatrix4x4 groundNormalMatrix;

    bool bAllocatedMemory;
    qreal retinaScale;
    float zooming_;
    QVector3D cameraPosition_;
    QVector3D cameraFocus_;
    QVector3D cameraUpDirection_;

    QVector2D lastMousePos;
    QVector3D translation_;
    QVector3D translationLag_;
    QVector3D rotation_;
    QVector3D rotationLag_;
    QVector3D scalingLag;
    SpecialKey specialKeyPressed;
    MouseButton mouseButtonPressed;

    ParticleViewMode currentParticleViewMode;
    ParticleColorMode currentParticleColorMode;
    MouseTransformationTarget currentMouseTransTarget;
    EnvironmentTexture currentEnvironmentTexture;
    FloorTexture currentFloorTexture;

    GLREAL* particleRandomColors;
    GLREAL* particleRampColors;
    GLREAL* particleSimulationColors;

    bool bInitializedScene;
    bool bTextureAnisotropicFiltering;
    bool enabledImageOutput;
    bool pausedImageOutput;
//    bool enabledClipYZPlane;
    QString imageOutputPath;
    QImage* outputImage;

    int current_frame_;

    bool isParticlesReady;
    bool bHideInvisibleParticles;
    int num_particles_;
    int num_sph_particles_;
    int num_pd_particles_;
    float sph_particle_radius_;
    float pd_particle_radius_;


    int viewportID_;


};

#endif // RENDERER_H
