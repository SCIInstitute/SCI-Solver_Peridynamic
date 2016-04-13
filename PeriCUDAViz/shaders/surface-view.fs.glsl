#version 410 core
//------------------------------------------------------------------------------------------
// fragment shader, phong shading
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
// uniforms
layout(std140) uniform Light
{
    vec4 position;
    vec4 color;
    float intensity;
} light;

layout(std140) uniform Material
{
    vec4 diffuseColor;
    vec4 specularColor;
    float reflection;
    float shininess;
} material;

uniform samplerCube envTex;
uniform sampler2D objTex;
uniform bool hasObjTex;

//------------------------------------------------------------------------------------------
// in variables
in VS_OUT
{
    vec3 f_color;
    vec3 f_normal;
    vec3 f_lightDir;
    vec3 f_viewDir;
    vec2 f_texcoord;
};

//----------------------------------------------------------`--------------------------------
// out variables
out vec4 fragColor;

//------------------------------------------------------------------------------------------
// const variables
const vec3 ambientLight = vec3(0.2);

//------------------------------------------------------------------------------------------
// If an object uses texture, it must set "GL_TRUE" to hasObjTex
// If it use vertex color, it must set material.diffuseColor.x to a number < 0.0f
//------------------------------------------------------------------------------------------
void main()
{
    vec3 normal = normalize(f_normal);
    vec3 lightDir = normalize(f_lightDir);
    vec3 viewDir = normalize(f_viewDir);
    vec3 reflectionDir = reflect(-viewDir, normal);

    float alpha = 1.0f;
    vec3 surfaceColor = vec3(0.0f);

    if(hasObjTex)
    {
        vec4 texVal = texture(objTex, f_texcoord);
        surfaceColor = texVal.xyz;
        alpha = texVal.w;
    }

    if(material.diffuseColor.x > -0.001f)
    {
        surfaceColor = mix(vec3(material.diffuseColor), surfaceColor, alpha);
    }
    else
    {
        surfaceColor = mix(f_color, surfaceColor, alpha);
    }

    vec3 ambient = ambientLight * surfaceColor;
    vec3 diffuse = vec3(max(dot(normal, lightDir), 0.0f)) * surfaceColor;

    vec3 halfDir = normalize(lightDir + viewDir);
    vec3 specular = pow(max(dot(halfDir, normal), 0.0f), material.shininess) * vec3(material.specularColor);

    vec3 reflection = vec3(0.0f);
    if(material.reflection > 0.0f)
    {
        reflection = texture(envTex, reflectionDir).xyz;
    }

    /////////////////////////////////////////////////////////////////
    // output
    fragColor = vec4(mix(light.intensity * (ambient + diffuse + specular), reflection, material.reflection), alpha);

}
