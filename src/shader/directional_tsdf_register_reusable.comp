#version 460
layout(local_size_x = 256) in;

// Registers a list of active-pool slots into the indexGrid (DirectionalTSDF design doc §10).
// Works for both reusable slots (classification output, Phase 2) and freshly-uploaded
// missing-group slots (Phase 1): each listed slot's meta must already hold its group key.

#define LOCAL_GRID 50
#define NUM_DIRS   6u

struct ActiveGroupMeta {
    int  gx;
    int  gy;
    int  gz;
    uint packed; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
};

layout(push_constant) uniform PC {
    uint g_count;
    int  g_baseX;
    int  g_baseY;
    int  g_baseZ;
};

layout(std430, set = 0, binding = 0) readonly buffer PoolIndexList { uint g_slots[]; };
layout(std430, set = 0, binding = 1) readonly buffer Meta { ActiveGroupMeta g_meta[]; };
layout(std430, set = 0, binding = 2) buffer IndexGrid { uint g_indexGrid[]; };

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= g_count) return;

    uint poolIndex = g_slots[k];
    ActiveGroupMeta m = g_meta[poolIndex];

    int lx = m.gx - g_baseX;
    int ly = m.gy - g_baseY;
    int lz = m.gz - g_baseZ;
    if (lx < 0 || ly < 0 || lz < 0 ||
        lx >= LOCAL_GRID || ly >= LOCAL_GRID || lz >= LOCAL_GRID)
        return;

    uint dir = m.packed & 0xFFu;
    uint cell = ((uint(lz) * uint(LOCAL_GRID) + uint(ly)) * uint(LOCAL_GRID) + uint(lx)) * NUM_DIRS + dir;
    g_indexGrid[cell] = poolIndex;
}
