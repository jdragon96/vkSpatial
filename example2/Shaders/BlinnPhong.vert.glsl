#version 450 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vViewPosition;
layout(location = 1) out vec3 vViewNormal;
layout(location = 2) out vec3 vColor;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 modelView;
} pc;

void main() {
    vec4 viewPosition = pc.modelView * vec4(inPosition, 1.0);
    vViewPosition = viewPosition.xyz;
    vViewNormal = normalize(mat3(pc.modelView) * inNormal);
    vColor = inColor;
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
