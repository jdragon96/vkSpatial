#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 1) in;

#define STACK_CAPACITY 128
#define MAX_K 64
#define INVALID_IDX 0xFFFFFFFFu
#define INF_DIST 1e38

layout(push_constant) uniform PC {
    float g_cx;
    float g_cy;
    float g_cz;
    uint g_k;
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
layout(std430, set = 0, binding = 5) writeonly buffer Distances {
    float g_distances[];
};
layout(std430, set = 0, binding = 6) buffer QueryState {
    uint g_unusedCount;
    uint g_status;
};

uint h_idx[MAX_K];
float h_dist[MAX_K];
uint h_size;

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

float aabbMinDist2(vec3 center, vec3 mn, vec3 mx) {
    vec3 delta = max(vec3(0.0), max(mn - center, center - mx));
    return dot(delta, delta);
}

bool worse(float distA, uint idxA, float distB, uint idxB) {
    return distA > distB || (distA == distB && idxA > idxB);
}

bool better(float distA, uint idxA, float distB, uint idxB) {
    return distA < distB || (distA == distB && idxA < idxB);
}

float heapWorst() {
    return h_size < g_k ? INF_DIST : h_dist[0];
}

void heapSiftUp(uint index) {
    while (index > 0u) {
        uint parent = (index - 1u) / 2u;
        if (!worse(
                h_dist[index], h_idx[index],
                h_dist[parent], h_idx[parent]))
            break;

        float distance = h_dist[parent];
        h_dist[parent] = h_dist[index];
        h_dist[index] = distance;
        uint primitive = h_idx[parent];
        h_idx[parent] = h_idx[index];
        h_idx[index] = primitive;
        index = parent;
    }
}

void heapSiftDown(uint index) {
    while (true) {
        uint left = index * 2u + 1u;
        uint right = left + 1u;
        uint worstIndex = index;

        if (left < h_size && worse(
                h_dist[left], h_idx[left],
                h_dist[worstIndex], h_idx[worstIndex]))
            worstIndex = left;
        if (right < h_size && worse(
                h_dist[right], h_idx[right],
                h_dist[worstIndex], h_idx[worstIndex]))
            worstIndex = right;
        if (worstIndex == index)
            break;

        float distance = h_dist[index];
        h_dist[index] = h_dist[worstIndex];
        h_dist[worstIndex] = distance;
        uint primitive = h_idx[index];
        h_idx[index] = h_idx[worstIndex];
        h_idx[worstIndex] = primitive;
        index = worstIndex;
    }
}

void heapInsert(uint primitive, float distance) {
    if (h_size < g_k) {
        h_idx[h_size] = primitive;
        h_dist[h_size] = distance;
        heapSiftUp(h_size);
        ++h_size;
    } else if (better(distance, primitive, h_dist[0], h_idx[0])) {
        h_idx[0] = primitive;
        h_dist[0] = distance;
        heapSiftDown(0u);
    }
}

void sortResults() {
    for (uint i = 1u; i < h_size; ++i) {
        uint primitive = h_idx[i];
        float distance = h_dist[i];
        uint j = i;
        while (j > 0u && better(
                distance, primitive,
                h_dist[j - 1u], h_idx[j - 1u])) {
            h_idx[j] = h_idx[j - 1u];
            h_dist[j] = h_dist[j - 1u];
            --j;
        }
        h_idx[j] = primitive;
        h_dist[j] = distance;
    }
}

void main() {
    vec3 center = vec3(g_cx, g_cy, g_cz);
    uint stackNode[STACK_CAPACITY];
    float stackDist[STACK_CAPACITY];
    uint top = 0u;
    h_size = 0u;
    g_status = 0u;

    stackNode[top] = 0u;
    stackDist[top] = 0.0;
    ++top;

    while (top > 0u) {
        --top;
        uint nodeIndex = stackNode[top];
        float nodeDistance = stackDist[top];
        if (nodeDistance > heapWorst())
            continue;

        WideNode node = g_nodes[nodeIndex];
        uint candidateNode[8];
        float candidateDist[8];
        uint candidateCount = 0u;

        for (uint slot = 0u; slot < node.childCount; ++slot) {
            vec3 childMin;
            vec3 childMax;
            decodeChild(node, slot, childMin, childMax);
            float childDistance =
                aabbMinDist2(center, childMin, childMax);
            if (childDistance > heapWorst())
                continue;

            if ((node.leafMask & (1u << slot)) != 0u) {
                LeafRange leaf = g_leaves[node.child[slot]];
                for (uint i = 0u; i < leaf.primitiveCount; ++i) {
                    uint sortedIndex = leaf.firstPrimitive + i;
                    Primitive prim =
                        g_primitives[g_mortonCodes[sortedIndex].index];
                    float primitiveDistance = aabbMinDist2(
                        center,
                        vec3(prim.aabbMinX, prim.aabbMinY, prim.aabbMinZ),
                        vec3(prim.aabbMaxX, prim.aabbMaxY, prim.aabbMaxZ));
                    heapInsert(prim.index, primitiveDistance);
                }
            } else {
                candidateNode[candidateCount] = node.child[slot];
                candidateDist[candidateCount] = childDistance;
                ++candidateCount;
            }
        }

        for (uint i = 1u; i < candidateCount; ++i) {
            uint nodeValue = candidateNode[i];
            float distanceValue = candidateDist[i];
            uint j = i;
            while (j > 0u && candidateDist[j - 1u] > distanceValue) {
                candidateNode[j] = candidateNode[j - 1u];
                candidateDist[j] = candidateDist[j - 1u];
                --j;
            }
            candidateNode[j] = nodeValue;
            candidateDist[j] = distanceValue;
        }

        for (int i = int(candidateCount) - 1; i >= 0; --i) {
            if (top == STACK_CAPACITY) {
                g_status = 1u;
                return;
            }
            stackNode[top] = candidateNode[i];
            stackDist[top] = candidateDist[i];
            ++top;
        }
    }

    sortResults();
    for (uint i = 0u; i < g_k; ++i) {
        g_results[i] = i < h_size ? h_idx[i] : INVALID_IDX;
        g_distances[i] = i < h_size ? h_dist[i] : INF_DIST;
    }
}
