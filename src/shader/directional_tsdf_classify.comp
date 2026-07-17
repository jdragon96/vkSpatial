#version 460
layout(local_size_x = 256) in;

// Classifies every active-pool slot against the new local base (design doc §9).
// Atomic-counter list building — POOL_CAPACITY ≈ 32K, contention is acceptable
// for the initial implementation (the doc says the same).

#define LOCAL_GRID 50

struct ActiveGroupMeta {
    int  gx;
    int  gy;
    int  gz;
    uint packed; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
};

layout(push_constant) uniform PC {
    uint g_poolCapacity;
    int  g_baseX;
    int  g_baseY;
    int  g_baseZ;
};

layout(std430, set = 0, binding = 0) readonly buffer Meta { ActiveGroupMeta g_meta[]; };
layout(std430, set = 0, binding = 1) buffer ReusableList { uint g_reusable[]; };
layout(std430, set = 0, binding = 2) buffer CleanFreeList { uint g_cleanFree[]; };
layout(std430, set = 0, binding = 3) buffer WriteBackList { uint g_writeBack[]; };
layout(std430, set = 0, binding = 4) buffer Counts { uint g_counts[3]; }; // reusable, cleanFree, writeBack

void main() {
    uint poolIndex = gl_GlobalInvocationID.x;
    if (poolIndex >= g_poolCapacity) return;

    ActiveGroupMeta m = g_meta[poolIndex];
    uint state = (m.packed >> 8) & 0xFFu;
    uint dirty = (m.packed >> 16) & 0xFFu;
    uint valid = (m.packed >> 24) & 0xFFu;

    // In-flight slots never enter any list (invariants #3/#4).
    if (state == 3u || state == 4u) return; // PendingUpload / PendingWriteBack

    if (valid == 0u || state == 0u) { // never used, or explicitly freed
        uint i = atomicAdd(g_counts[1], 1u);
        g_cleanFree[i] = poolIndex;
        return;
    }

    int lx = m.gx - g_baseX;
    int ly = m.gy - g_baseY;
    int lz = m.gz - g_baseZ;
    bool inside = lx >= 0 && ly >= 0 && lz >= 0 &&
                  lx < LOCAL_GRID && ly < LOCAL_GRID && lz < LOCAL_GRID;

    if (inside) {
        uint i = atomicAdd(g_counts[0], 1u);
        g_reusable[i] = poolIndex;
    } else if (dirty != 0u) {
        uint i = atomicAdd(g_counts[2], 1u);
        g_writeBack[i] = poolIndex;
    } else {
        uint i = atomicAdd(g_counts[1], 1u);
        g_cleanFree[i] = poolIndex;
    }
}
