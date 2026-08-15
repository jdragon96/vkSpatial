#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_count;
};

layout(std430, set = 0, binding = 0) readonly buffer Nodes {
    Node g_nodes[];
};

layout(std430, set = 0, binding = 1) readonly buffer ConstructionInfos {
    LBVHConstructionInfo g_construction[];
};

layout(std430, set = 0, binding = 2) buffer Ranges {
    BinaryRange g_ranges[];
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_count)
        return;

    uint leafOffset = g_count - 1u;
    uint leafIndex = leafOffset + gid;
    g_ranges[leafIndex] = BinaryRange(gid, 1u, 0u, 0u);
    memoryBarrierBuffer();

    if (g_count == 1u)
        return;

    uint nodeIndex = g_construction[leafIndex].parent;
    while (true) {
        uint previousVisits =
            atomicAdd(g_ranges[nodeIndex].visitationCount, 1u);
        if (previousVisits == 0u)
            return;

        memoryBarrierBuffer();

        Node node = g_nodes[nodeIndex];
        BinaryRange left = g_ranges[uint(node.left)];
        BinaryRange right = g_ranges[uint(node.right)];
        g_ranges[nodeIndex].firstPrimitive =
            min(left.firstPrimitive, right.firstPrimitive);
        g_ranges[nodeIndex].primitiveCount =
            left.primitiveCount + right.primitiveCount;

        memoryBarrierBuffer();
        if (nodeIndex == 0u)
            return;
        nodeIndex = g_construction[nodeIndex].parent;
    }
}
