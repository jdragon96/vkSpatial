#version 450

/// Pass 1 of 3. One thread per point: find (or create) the point's block record, then count the
/// point, its normal, and whether it was the first sample this frame in its coarse and fine cell.
///
/// Cell keys are exact, not hashed: a cell's coordinate WITHIN its block fits in 18 bits (fine) or
/// 15 (coarse), so (blockIndex, localCell) packs losslessly into 32 bits. Hashing the coordinate
/// instead would let collisions merge distinct cells, and that undercount would flow straight into
/// the occupancy ratio the whole decision rests on.

#include "voxel_common.glsl"                     // EMPTY_KEY, MAX_PROBE, wangHash
#include "DenseRegionClassifier.common.glsl"     // BlockRecord, NORMAL_FIXED_POINT_SCALE

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer Points  { float g_points[]; };
layout(std430, set = 0, binding = 1) readonly buffer Normals { float g_normals[]; };
layout(std430, set = 0, binding = 2) buffer Blocks     { BlockRecord g_blocks[]; };
layout(std430, set = 0, binding = 3) buffer BlockCount { uint g_blockCount; };
layout(std430, set = 0, binding = 4) buffer BlockIndex { uint g_blockIndex[]; };
layout(std430, set = 0, binding = 5) buffer FineCells   { uint g_fineCells[]; };
layout(std430, set = 0, binding = 6) buffer CoarseCells { uint g_coarseCells[]; };
// Points dropped because the block table's probing gave up before finding a slot. Must stay 0 in a
// healthy run -- a non-zero value means kBlockCapacity is too small for the scene.
layout(std430, set = 0, binding = 7) buffer BlockInsertFailures { uint g_blockInsertFailureCount; };

layout(push_constant) uniform PC
{
	uint  g_numPoints;
	uint  g_blockCapacity;
	uint  g_cellCapacity;
	float g_baseVoxel;
	int   g_blockVoxels;
};

/// Key-only open-addressed inserts. Each returns true when THIS call claimed the slot, which is
/// what makes "distinct cells this frame" countable without a second pass. Two near-identical
/// functions rather than one: GLSL has no reference-to-array parameter to pass the table with.

/// Disjoint 10-bit fields, spaced exactly 10 apart -- the XOR-with-overlapping-shifts form this
/// replaced aliased distinct blocks onto one key. Range is +/-512 blocks (about +/-163 m at 0.32 m
/// blocks); the maximum value is 0x3FFFFFFF, so a real key can never equal EMPTY_KEY.
uint packBlockKey(ivec3 block)
{
	return ((uint(block.x + 512) & 0x3FFu) << 20)
	     | ((uint(block.y + 512) & 0x3FFu) << 10)
	     |  (uint(block.z + 512) & 0x3FFu);
}

uint findOrInsertBlock(uint key)
{
	uint slot = wangHash(key) % g_blockCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_blockCapacity;
		uint previousKey = atomicCompSwap(g_blocks[index].blockKey, EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) { atomicAdd(g_blockCount, 1u); return index; }
		if (previousKey == key) return index;
	}
	return EMPTY_KEY;
}

bool claimFineCell(uint key)
{
	uint slot = wangHash(key) % g_cellCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_cellCapacity;
		uint previousKey = atomicCompSwap(g_fineCells[index], EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) return true;   // this thread claimed it -> count it
		if (previousKey == key) return false;        // already counted this frame
	}
	return false;
}

bool claimCoarseCell(uint key)
{
	uint slot = wangHash(key) % g_cellCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_cellCapacity;
		uint previousKey = atomicCompSwap(g_coarseCells[index], EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) return true;
		if (previousKey == key) return false;
	}
	return false;
}

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_numPoints) return;

	vec3 position = vec3(g_points[i * 3u], g_points[i * 3u + 1u], g_points[i * 3u + 2u]);
	vec3 normal   = vec3(g_normals[i * 3u], g_normals[i * 3u + 1u], g_normals[i * 3u + 2u]);

	// 1. Which block, and which cell within it, at both resolutions.
	float blockWorld = g_baseVoxel * float(g_blockVoxels);
	ivec3 block = ivec3(floor(position / blockWorld));
	vec3  local = position - vec3(block) * blockWorld;

	ivec3 coarseCell = clamp(ivec3(floor(local / g_baseVoxel)), ivec3(0), ivec3(g_blockVoxels - 1));
	ivec3 fineCell   = clamp(ivec3(floor(local / (g_baseVoxel * 0.5))), ivec3(0),
	                         ivec3(g_blockVoxels * 2 - 1));

	// 2. Block record.
	uint blockSlot = findOrInsertBlock(packBlockKey(block));
	if (blockSlot == EMPTY_KEY) {
		atomicAdd(g_blockInsertFailureCount, 1u); // table full -- drop is counted, not silent
		g_blockIndex[i] = EMPTY_KEY;
		return;
	}
	g_blockIndex[i] = blockSlot;

	// 3. Point and normal. The point count is kept both ways -- cumulative for the two ratio
	//    conditions, per-frame for the normal coherence, which needs the same frame's denominator
	//    as its numerator. `normal` is required to be unit length; see Record()'s contract.
	atomicAdd(g_blocks[blockSlot].pointCount, 1u);
	atomicAdd(g_blocks[blockSlot].pointCountFrame, 1u);
	atomicAdd(g_blocks[blockSlot].sumNormalFrameX, int(normal.x * NORMAL_FIXED_POINT_SCALE));
	atomicAdd(g_blocks[blockSlot].sumNormalFrameY, int(normal.y * NORMAL_FIXED_POINT_SCALE));
	atomicAdd(g_blocks[blockSlot].sumNormalFrameZ, int(normal.z * NORMAL_FIXED_POINT_SCALE));

	// 4. Occupancy, counted once per cell per frame.
	uint fineKey   = (blockSlot << 18) | (uint(fineCell.z) << 12)
	               | (uint(fineCell.y) << 6) | uint(fineCell.x);
	uint coarseKey = (blockSlot << 15) | (uint(coarseCell.z) << 10)
	               | (uint(coarseCell.y) << 5) | uint(coarseCell.x);

	if (claimFineCell(fineKey)) {
		atomicAdd(g_blocks[blockSlot].fineOccupied, 1u);      // cumulative, drives the ratio
		atomicAdd(g_blocks[blockSlot].fineOccupiedFrame, 1u); // this frame only, drives sizing
	}
	if (claimCoarseCell(coarseKey)) atomicAdd(g_blocks[blockSlot].coarseOccupied, 1u);
}
