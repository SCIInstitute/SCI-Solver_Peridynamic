#version 410 core
//------------------------------------------------------------------------------------------
// vertex shader, light shading
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

layout(std140) uniform Light
{
    vec4 position;
    vec4 color;
    float intensity;
} light;

uniform float pointDistance;
//------------------------------------------------------------------------------------------
void main()
{
    vec4 worldCoord = vec4(light.position.xyz, 1.0f);

    /////////////////////////////////////////////////////////////////
    // output
    gl_PointSize = 400.0f / pointDistance;
    gl_Position = projectionMatrix * viewMatrix * worldCoord;
}
