#version 450

#include "Realsense/Algorithm/Common.glsl"

/// *********************************************
/// Thin the emitted pixels down to one per detail voxel.
///
/// Two points closer together than one voxel are redundant to everything downstream -- a weighted
/// TSDF fusion averages them into the same cell, and an ICP correspondence search pays for both.
/// Removing one here removes it from the scatter's output, from the readback, and from every
/// consumer after that.
///
/// Measured on capture/ (8 frames, 1.95M points, scene median 0.82 m, f = 383): a 5 mm detail voxel
/// leaves 28.4% of the points and a 10 mm one leaves 8.4%. At 1.25 mm it leaves 99.1% -- the
/// lateral sample spacing is z/f, so there is nothing to remove unless z < detailVoxel * f, and the
/// C++ side leaves this pass off by default because of it.
///
/// TWO PASSES over one file, selected by -D. A global barrier has to separate them: every claim
/// must be settled before any pixel asks whether it won.
///
///   CLAIM  -- insert the voxel key and reduce the winner with atomicMin
///   CANCEL -- clear `emitted` on every pixel that is not its voxel's winner
///
/// Why atomicMin rather than first-wins: first-wins makes the survivor depend on which lane won a
/// race, and the compacted cloud's centroid is a float sum, so that order reaches the ICP solve.
/// This repository measured trajectories of 1.47, 6.45, 7.84 and 136.76 metres from four runs of
/// one command before the order leaks were closed. atomicMin always leaves the lowest pixel index
/// -- the row-major first one -- whatever order the lanes arrive in.
/// *********************************************

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	float g_detailVoxelMeters;
	uint  g_slotCount;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexGrid { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 1)          buffer ValidMask  { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2)          buffer VoxelTable { DownSampleSlot g_slots[]; };
layout(std430, set = 0, binding = 3)          buffer Counters   { DownSampleCounters g_counters; };

#define EMPTY_KEY 0xFFFFFFFFu
#define NO_WINNER 0xFFFFFFFFu
#define MAX_PROBE 32u

/// 11 bits for x and y, 10 for z.
///
/// In camera space z is always positive, so it needs no sign and no bias -- which is what buys the
/// eleventh bit for the two axes that do. At a 5 mm voxel this reaches +-5.1 m laterally and 5.1 m
/// deep, past a D435's usable range. At 1.25 mm it reaches only +-1.28 m, and the guard below sends
/// the rest through untouched; that resolution has nothing to remove anyway.
const int LATERAL_BIAS = 1024;

bool VoxelIsWithinPackableRange(ivec3 voxel)
{
	return voxel.x >= -LATERAL_BIAS && voxel.x < LATERAL_BIAS
	    && voxel.y >= -LATERAL_BIAS && voxel.y < LATERAL_BIAS
	    && voxel.z >= 0 && voxel.z < 1024;
}

uint PackVoxelKey(ivec3 voxel)
{
	return ((uint(voxel.x + LATERAL_BIAS) & 0x7FFu) << 21)
	     | ((uint(voxel.y + LATERAL_BIAS) & 0x7FFu) << 10)
	     |  (uint(voxel.z) & 0x3FFu);
}

/// Wang hash -- the same mixer TSDF's linear-probe table uses, so a key's slot is scattered rather
/// than clustered by the low bits of a coordinate that changes by one between neighbours.
uint WangHash(uint value)
{
	value = (value ^ 61u) ^ (value >> 16);
	value *= 9u;
	value = value ^ (value >> 4);
	value *= 0x27d4eb2du;
	value = value ^ (value >> 15);
	return value;
}

/// floor(), not truncation: truncation folds -0.4 and +0.4 into the same cell and doubles the
/// voxel that straddles the origin.
ivec3 VoxelOf(vec3 point)
{
	return ivec3(floor(point / g_detailVoxelMeters));
}

void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	uint pixel = uint(row * g_width + column);
	if (g_properties[pixel].emitted == 0u) return;

	ivec3 voxel = VoxelOf(g_vertices[pixel].xyz);

	// Out of the key's reach: keep the point. Fewer points removed is a throughput cost; a dropped
	// one is a hole in the surface.
	if (!VoxelIsWithinPackableRange(voxel))
	{
#ifdef DOWNSAMPLE_PASS_CLAIM
		atomicAdd(g_counters.outOfPackableRange, 1u);
#endif
		return;
	}

	uint key  = PackVoxelKey(voxel);
	uint slot = WangHash(key) % g_slotCount;

	for (uint probeStep = 0u; probeStep < MAX_PROBE; ++probeStep)
	{
		uint index = (slot + probeStep) % g_slotCount;

#ifdef DOWNSAMPLE_PASS_CLAIM
		// Claim an empty slot, or find the one this key already holds. Both outcomes then reduce
		// the winner; the compare-and-swap only decides WHERE the key lives, never who survives.
		uint previousKey = atomicCompSwap(g_slots[index].key, EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY || previousKey == key)
		{
			atomicMin(g_slots[index].winner, pixel);
			return;
		}
#else
		// CANCEL. The table is settled, so this is a plain read.
		uint found = g_slots[index].key;
		if (found == EMPTY_KEY) return; // the claim pass gave up on this pixel; it stays emitted
		if (found == key)
		{
			if (g_slots[index].winner != pixel) g_properties[pixel].emitted = 0u;
			return;
		}
#endif
	}

#ifdef DOWNSAMPLE_PASS_CLAIM
	// The probe budget ran out. The pixel keeps its flag and the ceiling is observable rather than
	// silent -- a table that is too small shows up here as a number, not as a slow frame.
	atomicAdd(g_counters.insertFailures, 1u);
#endif
}
