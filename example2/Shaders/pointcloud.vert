#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float pointSize;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 vColor;

void main() {
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    gl_PointSize = pc.pointSize;
    vColor = inColor;
}
