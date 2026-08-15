#version 450

#include "voxel_common.glsl"                     // EMPTY_KEY
#include "DenseRegionClassifier.common.glsl"     // BlockRecord

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer FineCells { uint g_fineCells[]; };
layout(std430, set = 0, binding = 1) buffer CoarseCells { uint g_coarseCells[]; };
layout(std430, set = 0, binding = 2) buffer Blocks { BlockRecord g_blocks[]; };

layout(push_constant) uniform PC { uint g_cellCapacity; uint g_blockCapacity; };

void main()
{
	uint i = gl_GlobalInvocationID.x;

	// 1. The two per-frame cell hashes.
	if (i < g_cellCapacity) {
		g_fineCells[i]   = EMPTY_KEY;
		g_coarseCells[i] = EMPTY_KEY;
	}

	// 2. Every per-frame field of the block record.
	if (i < g_blockCapacity) {
		g_blocks[i].pointCountFrame    = 0u;
		g_blocks[i].fineOccupiedFrame  = 0u;
		g_blocks[i].sumNormalFrameX    = 0;
		g_blocks[i].sumNormalFrameY    = 0;
		g_blocks[i].sumNormalFrameZ    = 0;
	}
}
