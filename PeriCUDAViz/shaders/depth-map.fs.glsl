#version 410 core
//------------------------------------------------------------------------------------------
// fragment shader, depth map shading
//------------------------------------------------------------------------------------------

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

    fragColor = gl_FragCoord.z * vec4(1, 1, 1, 1);
}
