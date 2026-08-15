#version 450

#include "voxel_common.glsl" // EMPTY_KEY

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer BlockIndex { uint g_blockIndex[]; };
layout(std430, set = 0, binding = 1) readonly buffer Dense      { uint g_dense[]; };
layout(std430, set = 0, binding = 2) buffer BaseIndex   { uint g_baseIndex[]; };
layout(std430, set = 0, binding = 3) buffer DetailIndex { uint g_detailIndex[]; };
layout(std430, set = 0, binding = 4) buffer Counts      { uint g_baseCount; uint g_detailCount; };

layout(push_constant) uniform PC { uint g_numPoints; };

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_numPoints) return;

	uint blockSlot = g_blockIndex[i];
	bool toDetail = (blockSlot != EMPTY_KEY) && (g_dense[blockSlot] != 0u);

	if (toDetail) g_detailIndex[atomicAdd(g_detailCount, 1u)] = i;
	else          g_baseIndex[atomicAdd(g_baseCount, 1u)] = i;
}
