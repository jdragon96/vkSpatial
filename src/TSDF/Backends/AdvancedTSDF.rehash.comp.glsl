#version 450

/// *********************************************************************************************
/// advanced_tsdf_rehash.comp.glsl
///
/// Re-insert every occupied slot of an OLD (smaller) hash into a NEW (larger, pre-cleared) hash.
/// Run when a tile's load factor gets high: the slot mapping is wangHash(key) % capacity, so a
/// resize needs a full re-hash, not a byte copy. Keys are unique across the table, so each old
/// entry finds its own slot; the accumulators + the per-slot first-fill frame move with it.
///
/// The new capacity is chosen so the resulting load stays low, so a free slot is always found well
/// within MAX_PROBE (no drops). g_filledCount is untouched -- the occupancy is unchanged by a rehash.
/// *********************************************************************************************

#include "voxel_common.glsl" // EMPTY_KEY, MAX_PROBE, wangHash

layout(local_size_x = 256) in;

/// Must match AdvDirEntry (AdvancedTSDF.h) / DirEntry in advanced_tsdf_integrate.comp.glsl.
struct DirEntry
{
	uint key;
	int  sumDW;
	uint sumW;
	int  sumNx;
	int  sumNy;
	int  sumNz;
};

layout(push_constant) uniform PC
{
	uint g_oldCapacity;
	uint g_newCapacity;
};

layout(std430, set = 0, binding = 0) readonly buffer OldHash { DirEntry g_old[]; };
layout(std430, set = 0, binding = 1) readonly buffer OldFirst { int g_oldFirst[]; };
layout(std430, set = 0, binding = 2) buffer NewHash { DirEntry g_new[]; };
layout(std430, set = 0, binding = 3) buffer NewFirst { int g_newFirst[]; };

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_oldCapacity)
		return;

	uint key = g_old[i].key;
	if (key == EMPTY_KEY)
		return;

	// Keys are unique across the old table, so no two threads race on the SAME key; atomicCompSwap
	// only resolves distinct-key collisions that probe onto a shared slot. Load is kept low, so the
	// free slot sits well within MAX_PROBE.
	uint slot = wangHash(key) % g_newCapacity;
	for (uint p = 0u; p < MAX_PROBE; ++p)
	{
		uint idx  = (slot + p) % g_newCapacity;
		uint prev = atomicCompSwap(g_new[idx].key, EMPTY_KEY, key);
		if (prev == EMPTY_KEY)
		{
			g_new[idx].sumDW = g_old[i].sumDW;
			g_new[idx].sumW  = g_old[i].sumW;
			g_new[idx].sumNx = g_old[i].sumNx;
			g_new[idx].sumNy = g_old[i].sumNy;
			g_new[idx].sumNz = g_old[i].sumNz;
			g_newFirst[idx]  = g_oldFirst[i];
			return;
		}
	}
}
