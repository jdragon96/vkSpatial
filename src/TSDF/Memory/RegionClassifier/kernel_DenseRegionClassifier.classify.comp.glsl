#version 450
#include "voxel_common.glsl"                     // EMPTY_KEY
#include "DenseRegionClassifier.common.glsl"     // BlockRecord, NORMAL_FIXED_POINT_SCALE

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) buffer Blocks { BlockRecord g_blocks[]; };
layout(std430, set = 0, binding = 1) buffer Dense  { uint g_dense[]; };
layout(std430, set = 0, binding = 2) buffer Totals { uint g_denseBlockCount; uint g_detailSlots; };

layout(push_constant) uniform PC
{
	uint  g_blockCapacity;
	float g_occupancyRatio;
	float g_normalCoherence;
	float g_samplesPerFineCell;
	uint  g_minimumFineOccupied;
};

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_blockCapacity) return;
	if (g_blocks[i].blockKey == EMPTY_KEY) return;

	atomicMax(g_blocks[i].fineOccupiedMax, g_blocks[i].fineOccupiedFrame);

	if (g_dense[i] != 0u) {                       // latched: never revert
		atomicAdd(g_denseBlockCount, 1u);
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
		return;
	}
	float pointCount      = float(g_blocks[i].pointCount);
	float pointCountFrame = float(g_blocks[i].pointCountFrame);
	float coarseOccupied  = float(g_blocks[i].coarseOccupied);
	float fineOccupied    = float(g_blocks[i].fineOccupied);
	if (coarseOccupied < 1.0 || pointCount < 1.0) return;
	if (pointCountFrame < 1.0) return;

	vec3 sumNormalFrame = vec3(float(g_blocks[i].sumNormalFrameX),
	                           float(g_blocks[i].sumNormalFrameY),
	                           float(g_blocks[i].sumNormalFrameZ)) / NORMAL_FIXED_POINT_SCALE;

	// 1. Spacing resolves the fine grid
	bool resolvesFineGrid = fineOccupied >= g_occupancyRatio * coarseOccupied;
	// 2. There is geometry detail to recover: normals disagree, so the region isn't flat.
	bool hasDetail        = length(sumNormalFrame) / pointCountFrame < g_normalCoherence;
	// 3. Enough samples survive the refinement for the finer voxels to be signal, not noise.
	bool keepsSignal      = pointCount >= g_samplesPerFineCell * fineOccupied;
	// 4. An absolute floor so a block glimpsed by a handful of points cannot latch to dense.
	bool hasSurface       = g_blocks[i].fineOccupiedFrame >= g_minimumFineOccupied;

	if (resolvesFineGrid && hasDetail && keepsSignal && hasSurface) {
		g_dense[i] = 1u;
		atomicAdd(g_denseBlockCount, 1u);
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
	}
}
