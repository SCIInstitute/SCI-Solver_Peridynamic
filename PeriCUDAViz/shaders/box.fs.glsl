#version 410 core
//------------------------------------------------------------------------------------------
// fragment shader, box rendering
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

uniform sampler2D objTex;
uniform bool hasObjTex;

uniform bool lineView;
//------------------------------------------------------------------------------------------
// const
const vec3 ambientLight = vec3(0.2);
//------------------------------------------------------------------------------------------
// in variables
in VS_OUT
{
    vec3 f_normal;
    vec3 f_lightDir;
    vec3 f_viewDir;
    vec2 f_texCoord;
};

//----------------------------------------------------------`--------------------------------
// out variables
out vec4 fragColor;

//------------------------------------------------------------------------------------------
// If an object uses texture, it must set "GL_TRUE" to hasObjTex
// If it use vertex color, it must set material.diffuseColor.x to a number < 0.0f
//------------------------------------------------------------------------------------------
void main()
{
    if(lineView)
    {
        fragColor = vec4(vec3(material.diffuseColor), 1.0);
    }
    else
    {
        vec3 normal = normalize(f_normal);
        vec3 lightDir = normalize(f_lightDir);
        vec3 viewDir = normalize(f_viewDir);
        vec3 reflectionDir = reflect(-viewDir, normal);

        float alpha = 0.0f;
        vec3 surfaceColor = vec3(0.0f);

        if(hasObjTex)
        {
            vec4 texVal = texture(objTex, f_texCoord);

            surfaceColor = texVal.xyz;
            alpha = texVal.w;
        }

        if(material.diffuseColor.x > -0.001f)
        {
            surfaceColor = mix(vec3(material.diffuseColor), surfaceColor, alpha);
        }
        else
        {
            surfaceColor = mix(vec3(0.8), surfaceColor, alpha);
        }

        vec3 ambient = vec3(0.0f);
        vec3 diffuse = vec3(0.0f);
        vec3 specular = vec3(0.0f);

        ambient = ambientLight * surfaceColor;
        diffuse = vec3(max(dot(normal, lightDir), 0.0f)) * surfaceColor;
        vec3 halfDir = normalize(lightDir + viewDir);
        specular = pow(max(dot(halfDir, normal), 0.0f), material.shininess) * vec3(material.specularColor);

        fragColor = vec4(ambient + light.intensity * (diffuse + specular), alpha);
    }
}
