#version 410 core
//------------------------------------------------------------------------------------------
// fragment shader, light shading
//------------------------------------------------------------------------------------------
layout(std140) uniform Light
{
    vec4 position;
    vec4 color;
    float intensity;
} light;

//uniform sampler2D pointTex;
//----------------------------------------------------------`--------------------------------
// out variables
out vec4 fragColor;

//------------------------------------------------------------------------------------------
void main()
{
    vec3 N;
//    texture(pointTex, gl_PointCoord);
    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle
    /////////////////////////////////////////////////////////////////
    // output
//    fragColor = vec4(vec3(light.intensity), 1.0f);
    fragColor = vec4(1.0f);

}
