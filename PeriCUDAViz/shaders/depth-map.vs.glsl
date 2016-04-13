#version 410 core
//------------------------------------------------------------------------------------------
// vertex shader, depth map rendering
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

//------------------------------------------------------------------------------------------
// in variables
in vec3 v_coord;

//------------------------------------------------------------------------------------------
void main()
{
    vec4 eyeCoord = viewMatrix * modelMatrix * vec4(vec3(v_coord), 1.0);
    vec3 posEye = vec3(eyeCoord);
    float dist = length(posEye);

    /////////////////////////////////////////////////////////////////
    // output
    gl_PointSize = pointRadius * (pointScale / dist);
    gl_Position = projectionMatrix * eyeCoord;
}
