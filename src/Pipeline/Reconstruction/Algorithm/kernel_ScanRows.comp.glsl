#version 450

// Serial exclusive scan in a single invocation.
//
// Deliberately not a parallel scan and deliberately not an atomicAdd append. The output of this
// chain is the ICP source cloud, whose centroid is a float sum -- so the ORDER of the compacted
// points reaches the solve, and a non-deterministic order makes a replay diverge. This repo has
// measured that: four runs of one command reported trajectories of 1.47, 6.45, 7.84 and 136.76
// metres before the order leaks were closed. One invocation over `height` entries (480 on a D435)
// is nothing next to the per-pixel passes, and it is deterministic by construction.
layout(local_size_x = 1) in;

layout(push_constant) uniform PC
{
	int g_height;
};

// In: per-row counts. Out: exclusive prefix sums, with the total in slot [g_height].
layout(std430, set = 0, binding = 0) buffer RowOffset { uint g_rowOffset[]; };

void main()
{
	if (gl_GlobalInvocationID.x != 0u) return;

	uint running = 0u;
	for (int row = 0; row < g_height; ++row)
	{
		uint count = g_rowOffset[row];
		g_rowOffset[row] = running;
		running += count;
	}
	g_rowOffset[g_height] = running;
}
