#version 460

#include "bvh_common.glsl"

layout(local_size_x = 64) in;

layout(push_constant) uniform Constants
{
    uint  g_numberOfPrimitive;

    float g_minX;
    float g_minY;
    float g_minZ;

    float g_maxX;
    float g_maxY;
    float g_maxZ;
};

layout(std430, set = 0, binding = 0) writeonly buffer MortonCodeBuffer
{
    MortonCode g_mortonCode[];
};

layout(std430, set = 0, binding = 1) readonly buffer Primitives
{
    Primitive g_primitives[];
};

// 10bit -> 30bit interleave
uint bitPadding(uint v)
{
    v &= 0x000003ffu;

    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;

    return v;
}

// 30bit morton code
uint point2morton(float x, float y, float z)
{
    const float SCALE = 1024.0f;
    const uint  MAX_COORD = 1023u;

    uint ix = uint(clamp(x * SCALE, 0.0f, float(MAX_COORD)));
    uint iy = uint(clamp(y * SCALE, 0.0f, float(MAX_COORD)));
    uint iz = uint(clamp(z * SCALE, 0.0f, float(MAX_COORD)));

    uint mx = bitPadding(ix);
    uint my = bitPadding(iy);
    uint mz = bitPadding(iz);

    return (mx << 2) | (my << 1) | mz;
}

void main()
{
    uint gid = gl_GlobalInvocationID.x;

    if (gid >= g_numberOfPrimitive)
        return;

    Primitive prim = g_primitives[gid];

    vec3 aabbMin = vec3(
        prim.aabbMinX,
        prim.aabbMinY,
        prim.aabbMinZ);

    vec3 aabbMax = vec3(
        prim.aabbMaxX,
        prim.aabbMaxY,
        prim.aabbMaxZ);

    // center = (min + max) * 0.5
    vec3 center = (aabbMin + aabbMax) * 0.5;
    vec3 g_min = vec3(g_minX, g_minY, g_minZ);
    vec3 g_max = vec3(g_maxX, g_maxY, g_maxZ);
    vec3 extent = max(g_max - g_min, vec3(1e-8));

    // normalize to [0,1]
    vec3 c = (center - g_min) / extent;
    g_mortonCode[gid] = MortonCode(point2morton(c.x, c.y, c.z), gid);
}