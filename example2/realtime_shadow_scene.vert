#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec3 vColor;
layout(location = 3) out vec4 vLightClip;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 lightDir;
} pc;

void main() {
    vec4 world = pc.model * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    vNormal = normalize(mat3(pc.model) * inNormal);
    vColor = inColor;
    vLightClip = pc.lightViewProj * world;
    gl_Position = pc.viewProj * world;
}
