#version 450

/// Pass 2 of 3. One thread per block record. Applies the four ANDed conditions and latches the
/// verdict: a block that has become dense never reverts, because a level that flips mid-scan
/// leaves a seam in the reconstruction.

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

	// Cumulative fineOccupied re-counts a cell once per frame, so only the per-frame value
	// estimates the slots the detail table actually needs. Updated every frame -- including ones
	// where the verdict below is already latched -- because accumulate wrote a fresh
	// fineOccupiedFrame this frame regardless of the block's dense/not-dense state.
	atomicMax(g_blocks[i].fineOccupiedMax, g_blocks[i].fineOccupiedFrame);

	if (g_dense[i] != 0u) {                       // latched: never revert
		atomicAdd(g_denseBlockCount, 1u);
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
		return;
	}

	// The two ratio conditions read the CUMULATIVE fields and the two single-viewpoint conditions
	// read THIS FRAME's, which is spec section 5's split, not an inconsistency: a ratio of two
	// cumulative counts is a coverage-weighted average of the per-frame ratios, which is exactly
	// what section 5 asks for, while spacing and curvature are single-viewpoint measurements that
	// overlapping viewpoints would otherwise manufacture out of registration error.
	float pointCount      = float(g_blocks[i].pointCount);
	float pointCountFrame = float(g_blocks[i].pointCountFrame);
	float coarseOccupied  = float(g_blocks[i].coarseOccupied);
	float fineOccupied    = float(g_blocks[i].fineOccupied);
	if (coarseOccupied < 1.0 || pointCount < 1.0) return;
	// A block this frame did not touch cannot newly latch, and dividing the normal sum by a zero
	// per-frame count would be a NaN comparison rather than a decision.
	if (pointCountFrame < 1.0) return;

	vec3 sumNormalFrame = vec3(float(g_blocks[i].sumNormalFrameX),
	                           float(g_blocks[i].sumNormalFrameY),
	                           float(g_blocks[i].sumNormalFrameZ)) / NORMAL_FIXED_POINT_SCALE;

	// 1. Spacing resolves the fine grid: occupiedFine/occupiedCoarse close to its ceiling of 4.
	bool resolvesFineGrid = fineOccupied >= g_occupancyRatio * coarseOccupied;
	// 2. There is geometry detail to recover: normals disagree, so the region isn't flat. This is
	//    the condition a density-only heuristic cannot express -- a dense flat plane passes every
	//    other test here and still has nothing worth a finer grid. Per frame: summed across frames
	//    it overflows int32 on exactly the flat wall it exists to veto (see the note in
	//    DenseRegionClassifier.common.glsl), and averaging normals across viewpoints mixes
	//    differently-misregistered estimates rather than sharpening one.
	bool hasDetail        = length(sumNormalFrame) / pointCountFrame < g_normalCoherence;
	// 3. Enough samples survive the refinement for the finer voxels to be signal, not noise.
	bool keepsSignal      = pointCount >= g_samplesPerFineCell * fineOccupied;
	// 4. An absolute floor so a block glimpsed by a handful of points cannot latch to dense.
	//    fineOccupiedFrame, not the cumulative fineOccupied: the cumulative field re-counts a cell
	//    once per frame, so a small block could otherwise cross this floor by being revisited rather
	//    than by having real extent -- exactly what this floor exists to rule out. Per-frame rather
	//    than fineOccupiedMax so that this and condition 2 describe the SAME frame: a block earns
	//    its detail level from one viewpoint that saw both the extent and the curvature, not from
	//    extent remembered off one frame and curvature measured on another.
	bool hasSurface       = g_blocks[i].fineOccupiedFrame >= g_minimumFineOccupied;

	if (resolvesFineGrid && hasDetail && keepsSignal && hasSurface) {
		g_dense[i] = 1u;
		atomicAdd(g_denseBlockCount, 1u);
		// fineOccupiedMax, not fineOccupied: the cumulative sum re-counts a cell once per frame,
		// so only the per-frame maximum estimates the slots the detail table actually needs.
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
	}
}
