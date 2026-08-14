#version 460

/// *********************************************
/// Constants
/// *********************************************
layout(local_size_x = 256) in;

#include "voxel_common.glsl"   // EMPTY_KEY

// 24 bytes. Layout must match DirEntry in advanced_tsdf_integrate.comp.glsl / AdvDirEntry.
struct DirEntry
{
	uint key;
	int  sumDW;
	uint sumW;
	int  sumNx;
	int  sumNy;
	int  sumNz;
};

layout(std430, set = 0, binding = 0) buffer HashTable { DirEntry g_hash[]; };

layout(push_constant) uniform PC { uint g_hashCapacity; };

/// *********************************************
/// Clear
/// *********************************************
// One thread per hash slot: reset it to EMPTY (key sentinel + zero accumulators). Replaces a per-tile
// host upload of a 24 MB "empty" buffer -- the dominant cost of creating a tile -- with a GPU write.
void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_hashCapacity) return;

	g_hash[i].key  = EMPTY_KEY;
	g_hash[i].sumDW = 0;
	g_hash[i].sumW  = 0u;
	g_hash[i].sumNx = 0;
	g_hash[i].sumNy = 0;
	g_hash[i].sumNz = 0;
}
