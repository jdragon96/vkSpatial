#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    // Round-point mask: discard the square point-sprite's corners outside radius 0.5.
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25)
        discard;
    outColor = vColor;
}
