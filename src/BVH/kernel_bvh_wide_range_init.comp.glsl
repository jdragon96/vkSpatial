#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_nodeCount;
};

layout(std430, set = 0, binding = 0) writeonly buffer Ranges {
    BinaryRange g_ranges[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_nodeCount)
        return;

    g_ranges[gid] = BinaryRange(0u, 0u, 0u, 0u);
}
