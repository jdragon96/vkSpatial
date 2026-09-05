#version 450
#include "Realsense/Algorithm/Common.glsl"

// Parallel stream compaction, one WORKGROUP per row.
//
// The output order is identical to a serial row walk, and that is the property this kernel exists
// to keep: the compacted cloud's centroid is a float sum, so the ORDER reaches the ICP solve, and a
// non-deterministic one makes a replay diverge -- this repository measured trajectories of 1.47,
// 6.45, 7.84 and 136.76 metres from four runs of one command before the order leaks were closed.
//
// What that constraint forbids is an atomicAdd append, whose slot depends on which lane won a
// race. It does NOT require a serial walk, which is what the first version of this kernel assumed:
// a slot computed as (row base + how many emitted pixels precede this column) is a pure function of
// position, so it is deterministic AND independent per pixel.
//
// Measured at 848x480, one invocation per row cost 0.351 ms -- more than twice the entire score
// kernel -- because 480 threads left the device idle and a warp's 64 lanes each read a different
// row, 6.8 KB apart, one cache line per lane per step.
//
// The C++ side must launch one workgroup PER ROW (Dispatch(height, 1, 1)), not DispatchElements
// over rows: gl_WorkGroupID.x IS the row here.
//
// Point, score and normal are scattered together. They have to be: the score is the fusion weight
// and the normal is what point-to-plane ICP solves against, and a compacted cloud whose weights and
// normals were left behind in image space cannot be re-paired with them -- the image coordinates
// are gone by the time anything downstream sees the array.

#define CHUNK 256

layout(local_size_x = CHUNK) in;

layout(push_constant) uniform PC
{
	int g_width;
	int g_height;
};

layout(std430, set = 0, binding = 0) readonly  buffer ValidMask     { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 1) readonly  buffer VertexGrid    { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 2) readonly  buffer RowOffset      { uint g_rowOffset[]; };
layout(std430, set = 0, binding = 3) writeonly buffer CompactPoints  { vec4 g_points[]; };
layout(std430, set = 0, binding = 4) writeonly buffer CompactScores  { float g_compactScores[]; };
layout(std430, set = 0, binding = 5) readonly  buffer NormalGrid     { vec4 g_normals[]; };
layout(std430, set = 0, binding = 6) writeonly buffer CompactNormals { vec4 g_compactNormals[]; };

shared uint s_flags[CHUNK];

// Emitted pixels of this row that precede the current chunk. Carried across chunks so a row wider
// than CHUNK still produces one contiguous, in-order run.
shared uint s_rowBase;

/// Hillis-Steele inclusive scan over s_flags.
///
/// Every barrier sits in uniform control flow: the bound is a compile-time constant and the caller
/// reaches this with the whole workgroup intact, so no lane can skip a barrier another waits on.
void ScanChunkInclusive(uint lane)
{
	for (uint offset = 1u; offset < uint(CHUNK); offset <<= 1u)
	{
		uint addend = (lane >= offset) ? s_flags[lane - offset] : 0u;
		barrier();
		s_flags[lane] += addend;
		barrier();
	}
}

/// slot = rowOffset[row] + |{ c < column : emitted(row, c) }|
///
/// slot      : where this pixel lands in the compacted array
/// rowOffset : exclusive prefix over ROWS, produced by ValidationMask.ScanRows.glsl
/// |{...}|   : exclusive prefix over COLUMNS within the row, computed here in shared memory
/// row       : gl_WorkGroupID.x, so it is uniform across the workgroup -- which is what lets the
///             early return below skip a whole workgroup without splitting one
void main()
{
	int row = int(gl_WorkGroupID.x);
	if (row >= g_height) return;

	uint lane = gl_LocalInvocationID.x;
	if (lane == 0u) s_rowBase = g_rowOffset[row];
	barrier();

	for (int chunkStart = 0; chunkStart < g_width; chunkStart += CHUNK)
	{
		int  column  = chunkStart + int(lane);
		uint emitted = 0u;
		if (column < g_width) emitted = g_properties[row * g_width + column].emitted;

		// 1. Scan the chunk. These reads are consecutive columns of ONE row, so a warp touches one
		//    run of cache lines -- the whole point of the rewrite.
		s_flags[lane] = emitted;
		barrier();
		ScanChunkInclusive(lane);

		// 2. Exclusive prefix is the inclusive one minus this lane's own flag.
		uint exclusive = s_flags[lane] - emitted;

		if (column < g_width && emitted != 0u)
		{
			uint slot  = s_rowBase + exclusive;
			int  pixel = row * g_width + column;
			// The score travels WITH the point. It is the fusion weight downstream, and a compacted
			// cloud whose confidences were left behind in image space cannot be re-paired with them.
			g_points[slot]         = g_vertices[pixel];
			g_compactScores[slot]  = g_properties[pixel].score;
			// The normal travels with the point for the same reason the score does.
			g_compactNormals[slot] = g_normals[pixel];
		}

		// 3. Carry this chunk's total forward. Read before the barrier that releases s_flags for
		//    the next chunk to overwrite.
		uint chunkTotal = s_flags[CHUNK - 1];
		barrier();
		if (lane == 0u) s_rowBase += chunkTotal;
		barrier();
	}
}
