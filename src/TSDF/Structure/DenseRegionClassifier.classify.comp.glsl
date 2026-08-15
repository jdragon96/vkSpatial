#version 450

/// Pass 2 of 3. One thread per block record. Applies the four ANDed conditions and latches the
/// verdict: a block that has become dense never reverts, because a level that flips mid-scan
/// leaves a seam in the reconstruction.

#include "voxel_common.glsl" // EMPTY_KEY

layout(local_size_x = 256) in;

struct BlockRecord
{
	uint blockKey;
	uint pointCount;
	uint coarseOccupied;
	uint fineOccupied;
	uint fineOccupiedFrame;
	uint fineOccupiedMax;
	int  sumNormalX;
	int  sumNormalY;
	int  sumNormalZ;
};

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

const float NORMAL_SCALE = 10000.0;

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

	float pointCount     = float(g_blocks[i].pointCount);
	float coarseOccupied = float(g_blocks[i].coarseOccupied);
	float fineOccupied   = float(g_blocks[i].fineOccupied);
	if (coarseOccupied < 1.0 || pointCount < 1.0) return;

	vec3 sumNormal = vec3(float(g_blocks[i].sumNormalX), float(g_blocks[i].sumNormalY),
	                      float(g_blocks[i].sumNormalZ)) / NORMAL_SCALE;

	// 1. Spacing resolves the fine grid: occupiedFine/occupiedCoarse close to its ceiling of 4.
	bool resolvesFineGrid = fineOccupied >= g_occupancyRatio * coarseOccupied;
	// 2. There is geometry detail to recover: normals disagree, so the region isn't flat. This is
	//    the condition a density-only heuristic cannot express -- a dense flat plane passes every
	//    other test here and still has nothing worth a finer grid.
	bool hasDetail        = length(sumNormal) / pointCount < g_normalCoherence;
	// 3. Enough samples survive the refinement for the finer voxels to be signal, not noise.
	bool keepsSignal      = pointCount >= g_samplesPerFineCell * fineOccupied;
	// 4. An absolute floor so a block glimpsed by a handful of points cannot latch to dense.
	bool hasSurface       = g_blocks[i].fineOccupied >= g_minimumFineOccupied;

	if (resolvesFineGrid && hasDetail && keepsSignal && hasSurface) {
		g_dense[i] = 1u;
		atomicAdd(g_denseBlockCount, 1u);
		// fineOccupiedMax, not fineOccupied: the cumulative sum re-counts a cell once per frame,
		// so only the per-frame maximum estimates the slots the detail table actually needs.
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
	}
}
