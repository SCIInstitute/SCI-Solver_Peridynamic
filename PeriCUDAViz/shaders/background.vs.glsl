#version 410 core
//------------------------------------------------------------------------------------------
// vertex shader, background shading
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

uniform vec3 cameraPosition;

//------------------------------------------------------------------------------------------
// in variables
in vec3 v_coord;

//------------------------------------------------------------------------------------------
// out variables
out vec3 f_viewDir;

//------------------------------------------------------------------------------------------
void main()
{
    vec4 worldCoord = modelMatrix * vec4(v_coord, 1.0);

    /////////////////////////////////////////////////////////////////
    // output
    f_viewDir = vec3(worldCoord) - vec3(cameraPosition) ;
    gl_Position = projectionMatrix * viewMatrix * worldCoord;
}
