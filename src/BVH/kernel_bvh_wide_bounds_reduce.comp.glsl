#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint g_count;
};

layout(std430, set = 0, binding = 0) readonly buffer Primitives {
    Primitive g_primitives[];
};

layout(std430, set = 0, binding = 1) buffer Bounds {
    SceneBounds g_bounds;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_count)
        return;

    Primitive prim = g_primitives[gid];
    atomicMin(g_bounds.minX, floatToOrdered(prim.aabbMinX));
    atomicMin(g_bounds.minY, floatToOrdered(prim.aabbMinY));
    atomicMin(g_bounds.minZ, floatToOrdered(prim.aabbMinZ));
    atomicMax(g_bounds.maxX, floatToOrdered(prim.aabbMaxX));
    atomicMax(g_bounds.maxY, floatToOrdered(prim.aabbMaxY));
    atomicMax(g_bounds.maxZ, floatToOrdered(prim.aabbMaxZ));
}
