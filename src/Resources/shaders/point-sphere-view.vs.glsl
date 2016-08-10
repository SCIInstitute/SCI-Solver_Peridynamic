#version 410 core
//------------------------------------------------------------------------------------------
// vertex shader, point-sphere-view
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// uniforms
layout(std140) uniform Matrices
{
    mat4 modelMatrix;
    mat4 normalMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
};


uniform float pointRadius;
uniform float pointScale;

uniform bool pointView;

uniform bool hideInvisibleParticles;

//------------------------------------------------------------------------------------------
in vec4 v_coord;
in vec3 v_color;
in int v_activity;
//------------------------------------------------------------------------------------------
out vec3 f_viewDir;
out vec3 f_color;

//------------------------------------------------------------------------------------------
const vec4 yzPlane = vec4(1, 0, 0, -1);

//------------------------------------------------------------------------------------------
void main()
{
//enum Activity
//{
//    ACTIVE = 0,
//    SEMI_ACTIVE,
//    INACTIVE,
//    INVISIBLE,
//    NUM_ACTIVITY_MODE
//};
    if(hideInvisibleParticles && v_activity == 3)
        return;

    vec4 eyeCoord = viewMatrix * modelMatrix * vec4(vec3(v_coord), 1.0);
    vec3 posEye = vec3(eyeCoord);
    float dist = length(posEye);

    /////////////////////////////////////////////////////////////////
    // output
    f_viewDir = posEye;
    f_color = v_color;

    if(pointView)
        gl_PointSize = 2.00;
    else
        gl_PointSize = pointRadius * (pointScale / dist);
    gl_Position = projectionMatrix * eyeCoord;
    gl_ClipDistance[0] = dot(modelMatrix * vec4(vec3(v_coord), 1.0), yzPlane);
}
