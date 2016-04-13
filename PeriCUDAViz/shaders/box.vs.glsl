#version 410 core
//------------------------------------------------------------------------------------------
// vertex shader, box rendering
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

uniform vec3 cameraPosition;

//------------------------------------------------------------------------------------------
// in variables
in vec3 v_coord;
in vec3 v_normal;
in vec2 v_texCoord;

//------------------------------------------------------------------------------------------
// out variables
out VS_OUT
{
    vec3 f_normal;
    vec3 f_lightDir;
    vec3 f_viewDir;
    vec2 f_texCoord;
};

//------------------------------------------------------------------------------------------
void main()
{
    vec4 worldCoord = modelMatrix * vec4(v_coord, 1.0);

    /////////////////////////////////////////////////////////////////
    // output
    f_normal = mat3(normalMatrix) * v_normal;
    f_lightDir = vec3(light.position) - vec3(worldCoord);
    f_viewDir = vec3(cameraPosition) - vec3(worldCoord);
    f_texCoord = v_texCoord;

    gl_Position = projectionMatrix * viewMatrix * worldCoord;
}
