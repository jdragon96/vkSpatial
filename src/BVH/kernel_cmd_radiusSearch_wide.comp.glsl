#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 1) in;

#define STACK_CAPACITY 128

layout(push_constant) uniform PC {
    float g_cx;
    float g_cy;
    float g_cz;
    float g_radius;
    uint g_maxResults;
};

layout(std430, set = 0, binding = 0) readonly buffer Nodes {
    WideNode g_nodes[];
};
layout(std430, set = 0, binding = 1) readonly buffer Leaves {
    LeafRange g_leaves[];
};
layout(std430, set = 0, binding = 2) readonly buffer MortonCodes {
    MortonCode g_mortonCodes[];
};
layout(std430, set = 0, binding = 3) readonly buffer Primitives {
    Primitive g_primitives[];
};
layout(std430, set = 0, binding = 4) writeonly buffer Results {
    uint g_results[];
};
layout(std430, set = 0, binding = 5) buffer QueryState {
    uint g_resultCount;
    uint g_status;
};

uint unpackByte(uint word, uint slot) {
    return (word >> ((slot & 3u) * 8u)) & 0xFFu;
}

void decodeChild(in WideNode node, uint slot, out vec3 mn, out vec3 mx) {
    uint word = slot / 4u;
    vec3 qMin = vec3(
        unpackByte(node.qBounds[0u + word], slot),
        unpackByte(node.qBounds[2u + word], slot),
        unpackByte(node.qBounds[4u + word], slot));
    vec3 qMax = vec3(
        unpackByte(node.qBounds[6u + word], slot),
        unpackByte(node.qBounds[8u + word], slot),
        unpackByte(node.qBounds[10u + word], slot));
    vec3 origin = vec3(node.originX, node.originY, node.originZ);
    vec3 scale = vec3(node.scaleX, node.scaleY, node.scaleZ);
    mn = origin + scale * qMin;
    mx = origin + scale * qMax;
}

bool sphereAABBIntersect(vec3 center, float radius, vec3 mn, vec3 mx) {
    vec3 closest = clamp(center, mn, mx);
    vec3 delta = closest - center;
    return dot(delta, delta) <= radius * radius;
}

void main() {
    vec3 center = vec3(g_cx, g_cy, g_cz);
    uint stack[STACK_CAPACITY];
    uint top = 0u;
    uint count = 0u;

    g_resultCount = 0u;
    g_status = 0u;
    stack[top++] = 0u;

    while (top > 0u) {
        WideNode node = g_nodes[stack[--top]];

        for (uint slot = 0u; slot < node.childCount; ++slot) {
            vec3 childMin;
            vec3 childMax;
            decodeChild(node, slot, childMin, childMax);
            if (!sphereAABBIntersect(center, g_radius, childMin, childMax))
                continue;

            if ((node.leafMask & (1u << slot)) != 0u) {
                LeafRange leaf = g_leaves[node.child[slot]];
                for (uint i = 0u; i < leaf.primitiveCount; ++i) {
                    uint sortedIndex = leaf.firstPrimitive + i;
                    Primitive prim =
                        g_primitives[g_mortonCodes[sortedIndex].index];
                    vec3 primMin = vec3(
                        prim.aabbMinX, prim.aabbMinY, prim.aabbMinZ);
                    vec3 primMax = vec3(
                        prim.aabbMaxX, prim.aabbMaxY, prim.aabbMaxZ);
                    if (!sphereAABBIntersect(
                            center, g_radius, primMin, primMax))
                        continue;

                    if (count < g_maxResults)
                        g_results[count] = prim.index;
                    ++count;
                }
            } else {
                if (top == STACK_CAPACITY) {
                    g_resultCount = count;
                    g_status = 1u;
                    return;
                }
                stack[top++] = node.child[slot];
            }
        }
    }

    g_resultCount = count;
}
