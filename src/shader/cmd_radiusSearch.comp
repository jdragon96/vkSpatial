#version 460
#extension GL_GOOGLE_include_directive: enable
#include "bvh_common.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    float    g_cx;
    float    g_cy;
    float    g_cz;
    float    g_radius;
    uint     g_maxResults;
};

layout(std430, set = 0, binding = 0) readonly buffer Nodes {
    Node g_nodes[];
};

layout(std430, set = 0, binding = 1) writeonly buffer Results {
    uint g_results[];
};

layout(std430, set = 0, binding = 2) buffer ResultCount {
    uint g_resultCount;
};

bool sphereAABBIntersect(vec3 c, float r, vec3 mn, vec3 mx) {
    vec3 closest = clamp(c, mn, mx);
    vec3 d = closest - c;
    return dot(d, d) <= r * r;
}

void main() {
    vec3 center = vec3(g_cx, g_cy, g_cz);
    g_resultCount = 0u;

    int stack[64];
    int top = 0;
    stack[top++] = 0; // root

    uint count = 0u;
    while (top > 0) {
        int nodeIdx = stack[--top];
        Node node = g_nodes[nodeIdx];

        if (!sphereAABBIntersect(
                center, 
                g_radius,
                vec3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
                vec3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ)))
            continue;

        if (node.left == INVALID_POINTER) {
            if (count < g_maxResults)
                g_results[count] = node.primitiveIdx;
            count++;
        } else {
            if (top + 2 <= 64) {
                stack[top++] = node.right;
                stack[top++] = node.left;
            }
        }
    }

    g_resultCount = count;
}
