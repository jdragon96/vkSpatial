#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 1) in;

layout(std430, set = 0, binding = 0) writeonly buffer Queue {
    uint g_queue[];
};

layout(std430, set = 0, binding = 1) writeonly buffer State {
    WideBuildState g_state;
};

void main() {
    g_queue[0] = 0u;
    g_state = WideBuildState(1u, 0u, 0u, 0u);
}
