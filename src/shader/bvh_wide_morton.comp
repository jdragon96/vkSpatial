#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    uint g_count;
};

layout(std430, set = 0, binding = 0) writeonly buffer MortonCodes {
    MortonCode g_mortonCodes[];
};

layout(std430, set = 0, binding = 1) readonly buffer Primitives {
    Primitive g_primitives[];
};

layout(std430, set = 0, binding = 2) readonly buffer Bounds {
    SceneBounds g_bounds;
};

uint bitPadding(uint value) {
    value &= 0x000003FFu;
    value = (value | (value << 16u)) & 0x030000FFu;
    value = (value | (value << 8u)) & 0x0300F00Fu;
    value = (value | (value << 4u)) & 0x030C30C3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

uint pointToMorton(vec3 point) {
    uvec3 coordinate = uvec3(clamp(point * 1024.0, 0.0, 1023.0));
    return (bitPadding(coordinate.x) << 2u) |
           (bitPadding(coordinate.y) << 1u) |
           bitPadding(coordinate.z);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_count)
        return;

    Primitive prim = g_primitives[gid];
    vec3 center = 0.5 * (
        vec3(prim.aabbMinX, prim.aabbMinY, prim.aabbMinZ) +
        vec3(prim.aabbMaxX, prim.aabbMaxY, prim.aabbMaxZ));
    vec3 sceneMin = vec3(
        orderedToFloat(g_bounds.minX),
        orderedToFloat(g_bounds.minY),
        orderedToFloat(g_bounds.minZ));
    vec3 sceneMax = vec3(
        orderedToFloat(g_bounds.maxX),
        orderedToFloat(g_bounds.maxY),
        orderedToFloat(g_bounds.maxZ));
    vec3 extent = max(sceneMax - sceneMin, vec3(1e-20));
    vec3 normalized = clamp((center - sceneMin) / extent, 0.0, 1.0);

    g_mortonCodes[gid] = MortonCode(pointToMorton(normalized), gid);
}
