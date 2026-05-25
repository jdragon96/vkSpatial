#version 460
#extension GL_GOOGLE_include_directive: enable
#include "bvh_common.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    float  g_cx;
    float  g_cy;
    float  g_cz;
    uint   g_k;
};

#define MAX_K       64
#define INVALID_IDX 0xFFFFFFFFu
#define INF_DIST    1e38

layout(std430, set = 0, binding = 0) readonly buffer Nodes    { Node   g_nodes[];    };
layout(std430, set = 0, binding = 1) writeonly buffer Results { uint   g_results[];  };
layout(std430, set = 0, binding = 2) writeonly buffer Dists   { float  g_distances[]; };

// per-invocation max-heap: root = worst (farthest) neighbor
uint  h_idx[MAX_K];
float h_dist[MAX_K];
uint  h_size;

// squared distance from point c to AABB [mn, mx]
// for PointPrim (mn == mx) this equals |p - c|^2
float aabbMinDist2(vec3 c, vec3 mn, vec3 mx) {
    vec3 d = max(vec3(0.0), max(mn - c, c - mx));
    return dot(d, d);
}

float heapWorst() {
    return (h_size < g_k) ? INF_DIST : h_dist[0];
}

void heapSiftUp(uint i) {
    while (i > 0u) {
        uint p = (i - 1u) / 2u;
        if (h_dist[p] < h_dist[i]) {
            float td = h_dist[p]; h_dist[p] = h_dist[i]; h_dist[i] = td;
            uint  ti = h_idx[p];  h_idx[p]  = h_idx[i];  h_idx[i]  = ti;
            i = p;
        } else break;
    }
}

void heapSiftDown(uint i) {
    while (true) {
        uint l = 2u * i + 1u, r = 2u * i + 2u, m = i;
        if (l < h_size && h_dist[l] > h_dist[m]) m = l;
        if (r < h_size && h_dist[r] > h_dist[m]) m = r;
        if (m == i) break;
        float td = h_dist[i]; h_dist[i] = h_dist[m]; h_dist[m] = td;
        uint  ti = h_idx[i];  h_idx[i]  = h_idx[m];  h_idx[m]  = ti;
        i = m;
    }
}

void heapInsert(uint idx, float dist) {
    if (h_size < g_k) {
        h_idx[h_size]  = idx;
        h_dist[h_size] = dist;
        heapSiftUp(h_size);
        h_size++;
    } else if (dist < h_dist[0]) {
        // replace worst neighbor and restore heap property
        h_idx[0]  = idx;
        h_dist[0] = dist;
        heapSiftDown(0u);
    }
}

void main() {
    vec3 center = vec3(g_cx, g_cy, g_cz);
    h_size = 0u;

    int stack[64];
    int top = 0;
    stack[top++] = 0; // root

    while (top > 0) {
        int  nodeIdx = stack[--top];
        Node node    = g_nodes[nodeIdx];

        float minD2 = aabbMinDist2(center,
            vec3(node.aabbMinX, node.aabbMinY, node.aabbMinZ),
            vec3(node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ));

        // prune: this subtree cannot contain a closer neighbor than current worst
        if (minD2 > heapWorst()) continue;

        if (node.left == INVALID_POINTER) {
            // leaf: minD2 == actual squared distance for PointPrim (mn == mx)
            heapInsert(node.primitiveIdx, minD2);
        } else {
            if (top + 2 <= 64) {
                stack[top++] = node.right;
                stack[top++] = node.left;
            }
        }
    }

    for (uint i = 0u; i < g_k; i++) {
        g_results[i]   = (i < h_size) ? h_idx[i]  : INVALID_IDX;
        g_distances[i] = (i < h_size) ? h_dist[i] : INF_DIST;
    }
}
