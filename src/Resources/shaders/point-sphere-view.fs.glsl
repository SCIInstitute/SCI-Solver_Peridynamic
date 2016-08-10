#version 410 core
//------------------------------------------------------------------------------------------
// fragment shader, point-sphere-view
//------------------------------------------------------------------------------------------
layout(std140) uniform Light
{
    vec4 direction;
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

uniform bool hasVertexColor;
//------------------------------------------------------------------------------------------
// const variables
const vec3 ambientLight = vec3(0.3);

//----------------------------------------------------------`--------------------------------
in vec3 f_viewDir;
in vec3 f_color;
//----------------------------------------------------------`--------------------------------
// out variables
out vec4 fragColor;

//------------------------------------------------------------------------------------------
void main()
{
    vec3 viewDir = normalize(f_viewDir);
    vec3 lightDir = normalize(vec3(light.direction));
    vec3 N;
//    texture(pointTex, gl_PointCoord);
    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    vec3 surfaceColor = vec3(material.diffuseColor);
    if(hasVertexColor)
        surfaceColor = f_color;

    vec3 ambient = ambientLight * surfaceColor;
    vec3 diffuse = max(dot(N, lightDir), 0.0f) * surfaceColor;

    vec3 halfDir = normalize(lightDir + viewDir);
    vec3 specular = pow(max(dot(halfDir, N), 0.0f), material.shininess) * vec3(material.specularColor);

    /////////////////////////////////////////////////////////////////
    // output
    fragColor = vec4(ambient + light.intensity *(diffuse + specular), 1.0f);

}
