#version 460
#extension GL_GOOGLE_include_directive: enable

#include "bvh_wide_common.glsl"

layout(local_size_x = 1) in;

layout(std430, set = 0, binding = 0) writeonly buffer Bounds {
    SceneBounds g_bounds;
};

void main() {
    uint positiveInfinity = floatToOrdered(uintBitsToFloat(0x7F800000u));
    uint negativeInfinity = floatToOrdered(uintBitsToFloat(0xFF800000u));

    g_bounds = SceneBounds(
        positiveInfinity,
        positiveInfinity,
        positiveInfinity,
        negativeInfinity,
        negativeInfinity,
        negativeInfinity);
}
