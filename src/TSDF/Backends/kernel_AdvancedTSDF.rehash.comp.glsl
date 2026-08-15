#version 450

/// *********************************************************************************************
/// kernel_AdvancedTSDF.rehash.comp.glsl
///
/// Re-insert every occupied slot of an OLD (smaller) hash into a NEW (larger, pre-cleared) hash,
/// through findOrInsert -- the SAME addressing the integrate kernel uses for whichever hash
/// strategy this pipeline was Define()'d with. A hand-rolled linear probe here (as this kernel
/// used to have) would insert every entry at wangHash(key) % newCapacity regardless of strategy;
/// a bucketed table addresses by wangHash(key) % bucketCount instead, so those entries would land
/// outside every bucket a later bucketed findSlot/findOrInsert would ever probe for that key --
/// invisible to lookup, silently duplicated by the next integrate. Run when a tile's load factor
/// gets high: the slot mapping depends on capacity, so a resize needs a full re-hash, not a byte
/// copy. Keys are unique across the table, so each old entry finds its own slot; the accumulators
/// + the per-slot first-fill frame move with it.
///
/// The new capacity is chosen so the resulting load stays low, so a free slot is always found well
/// within MAX_PROBE (no drops).
/// *********************************************************************************************

#include "voxel_common.glsl" // EMPTY_KEY, MAX_PROBE, wangHash

layout(local_size_x = 256) in;

/// Must match AdvDirEntry (AdvancedTSDF.h) / DirEntry in kernel_AdvancedTSDF.integrate.comp.glsl.
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
	uint g_hashCapacity; // the NEW table's capacity -- named for the hash contract, findOrInsert reads it
	// findOrInsert stamps g_firstFrame[slot] with this the moment it claims a slot, but every claim
	// here is immediately overwritten below with the OLD entry's real first-fill frame -- so this
	// value is never actually read back. Any value works; it is a named field rather than a magic
	// 0 in the C++ Args() call so a reader does not have to guess why it goes unused.
	int  g_currentFrame;
};

layout(std430, set = 0, binding = 0) readonly buffer OldHash { DirEntry g_old[]; };
layout(std430, set = 0, binding = 1) readonly buffer OldFirst { int g_oldFirst[]; };
// Bound under the hash contract's own names (g_hash / g_hashCapacity / g_firstFrame /
// g_filledCount / g_currentFrame) so findOrInsert below addresses the NEW table exactly as the
// integrate kernel would -- linear or bucketed, whichever HASH_* macro this pipeline was built with.
layout(std430, set = 0, binding = 2) buffer HashTable { DirEntry g_hash[]; };
layout(std430, set = 0, binding = 3) buffer FirstFrame { int g_firstFrame[]; };
// Scratch: a rehash MOVES entries, it creates none, so the real occupancy counter must not see
// these claims. This binding is a throwaway the caller never reads back or resets.
layout(set = 0, binding = 4) buffer Stat { uint g_filledCount; };

#define HASH_WITH_INSERT
#include "TSDF/Memory/Hash/HashStrategy.glsl"

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
	uint slot = findOrInsert(key);
	if (slot == HASH_INSERT_FAILED)
		return; // unchanged from before: the entry is silently dropped, same as a failed probe always was

	g_hash[slot].sumDW = g_old[i].sumDW;
	g_hash[slot].sumW  = g_old[i].sumW;
	g_hash[slot].sumNx = g_old[i].sumNx;
	g_hash[slot].sumNy = g_old[i].sumNy;
	g_hash[slot].sumNz = g_old[i].sumNz;
	g_firstFrame[slot] = g_oldFirst[i]; // overwrites findOrInsert's g_currentFrame stamp
}
