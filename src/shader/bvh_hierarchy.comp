#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_common.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_count;
    uint g_absolute_pointers;
};

layout(std430, set = 0, binding = 0) readonly buffer SortedMortonCodes {
    MortonCode g_mortonCodes[];
};

layout(std430, set = 0, binding = 1) readonly buffer Primitives {
    Primitive g_primitives[];
};

layout(std430, set = 0, binding = 2) writeonly buffer Nodes {
    Node g_nodes[];
};

layout(std430, set = 0, binding = 3) buffer ConstructionInfos {
    LBVHConstructionInfo g_constrInfos[];
};

int nearest(int i, uint codeI, int j) {
    if (j < 0 || j > int(g_count) - 1) {
        return -1;
    }
    uint codeJ = g_mortonCodes[j].code;
    if (codeI == codeJ) {
        return 32 + 31 - int(findMSB(uint(i) ^ uint(j)));
    }
    return 31 - int(findMSB(codeI ^ codeJ));
}

void determineRange(int idx, out int lower, out int upper) {
    const uint code  = g_mortonCodes[idx].code;
    const int nearestLeft = nearest(idx, code, idx - 1);
    const int nearestRight = nearest(idx, code, idx + 1);
    const int d      = (nearestRight >= nearestLeft) ? 1 : -1;
    const int minimumNearest = min(nearestLeft, nearestRight);

    // find boundary using binary search
    int outIndex = 2;
    while (nearest(idx, code, idx + outIndex * d) > minimumNearest) {
        outIndex <<= 1;
    }

    int cursor = 0;
    for (int p = outIndex >> 1; p > 0; p >>= 1){
        if (nearest(idx, code, idx + (cursor + p) * d) > minimumNearest) {
            cursor += p;
        }
    }

    int jdx = idx + cursor * d;
    lower = min(idx, jdx);
    upper = max(idx, jdx);
}

int findSplit(int first, int last) {
    uint firstCode    = g_mortonCodes[first].code;
    int  commonPrefix = nearest(first, firstCode, last);

    int low  = first;
    int high = last - 1;
    int split = first;

    while (low <= high) {
        int mid = (low + high) / 2;

        if (nearest(first, firstCode, mid) > commonPrefix) {
            split = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return split;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    const int LEAF_OFFSET = int(g_count) - 1;
    
    // 1. update default leaft node
    if (gid < g_count) {
        Primitive prim = g_primitives[g_mortonCodes[gid].index];
        g_nodes[LEAF_OFFSET + int(gid)] = Node(
            INVALID_POINTER, INVALID_POINTER,
            prim.index,
            prim.aabbMinX, prim.aabbMinY, prim.aabbMinZ,
            prim.aabbMaxX, prim.aabbMaxY, prim.aabbMaxZ
        );
    }

    // 2. update internal node
    if (gid < g_count - 1u) {
        int first, last;
        determineRange(int(gid), first, last);

        int split = findSplit(first, last);

        bool isLeftChildLeaf = (split == first);
        int childA = isLeftChildLeaf ? (LEAF_OFFSET + split) : split;

        bool isRightChildLeaf = (split + 1 == last);
        int childB = isRightChildLeaf ? (LEAF_OFFSET + split + 1) : split + 1;

        if (g_absolute_pointers != 0u) {
            g_nodes[gid] = Node(childA, childB, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        } else {
            int offsetA = childA - int(gid);
            int offsetB = childB - int(gid);
            g_nodes[gid] = Node(offsetA, offsetB, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        g_constrInfos[childA] = LBVHConstructionInfo(uint(gid), 0);
        g_constrInfos[childB] = LBVHConstructionInfo(uint(gid), 0);
    }

    // 3. update root node
    if (gid == 0u) {
        g_constrInfos[0] = LBVHConstructionInfo(0u, 0);
    }
}
