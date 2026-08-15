#version 450

/// Empties both per-frame cell hashes. They are cleared every frame because keeping them across
/// frames would mean remembering every distinct fine cell ever seen -- the same order of memory as
/// the detail TSDF this classifier is deciding about.

#include "voxel_common.glsl"                     // EMPTY_KEY
#include "DenseRegionClassifier.common.glsl"     // BlockRecord

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer FineCells { uint g_fineCells[]; };
layout(std430, set = 0, binding = 1) buffer CoarseCells { uint g_coarseCells[]; };
layout(std430, set = 0, binding = 2) buffer Blocks { BlockRecord g_blocks[]; };

layout(push_constant) uniform PC { uint g_cellCapacity; uint g_blockCapacity; };

// Dispatched over max(cellCapacity, blockCapacity); each guard covers its own range. Resetting the
// per-frame occupancy counter here -- rather than in a fourth dispatch -- keeps the frame's setup
// to one kernel.
void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i < g_cellCapacity) {
		g_fineCells[i]   = EMPTY_KEY;
		g_coarseCells[i] = EMPTY_KEY;
	}
	if (i < g_blockCapacity) g_blocks[i].fineOccupiedFrame = 0u;
}
