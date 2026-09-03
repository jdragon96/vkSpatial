#version 450
#include "ValidationMask.common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	float g_relativeDepthJump;
	float g_minimumDepthJump;
	uint  g_symmetricDepthJumpGuard;  // [H4] 0 = off
	int   g_minimumValidNeighbours;   // [H5] 0 = off
	float g_minimumIncidenceCosine;   // [H6] 0 = off, else cos(maximumIncidenceDegrees)
};

layout(std430, set = 0, binding = 0) readonly buffer VertexGrid { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 1) buffer ValidMask { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2) writeonly buffer NormalGrid { vec4 g_normals[]; };
layout(std430, set = 0, binding = 3) buffer Counters { ValidationMaskCounters g_counters; };

/// [H4] Does any of the eight neighbours exist and lie further than `tolerance` in depth?
///
/// A neighbour outside the image is absent, not a step, so the image border is never rejected for
/// having one.
bool StraddlesADepthStep(int column, int row, float depth, float tolerance)
{
	for (int deltaRow = -1; deltaRow <= 1; ++deltaRow) {
		int neighbourRow = row + deltaRow;
		if (neighbourRow < 0 || neighbourRow >= g_height) continue;

		for (int deltaColumn = -1; deltaColumn <= 1; ++deltaColumn) {
			if (deltaRow == 0 && deltaColumn == 0) continue;
			int neighbourColumn = column + deltaColumn;
			if (neighbourColumn < 0 || neighbourColumn >= g_width) continue;

			int neighbour = neighbourRow * g_width + neighbourColumn;
			if (g_properties[neighbour].valid == 0u) continue;
			if (abs(g_vertices[neighbour].z - depth) > tolerance) return true;
		}
	}
	return false;
}

/// [H5] How many of the eight neighbours are valid AND on the same surface.
///
/// "Same surface" reuses the jump tolerance rather than counting mere validity: a 2x2 blob of near
/// surface on a far wall has eight valid neighbours and only three of its own.
int CountSameSurfaceNeighbours(int column, int row, float depth, float tolerance)
{
	int count = 0;
	for (int deltaRow = -1; deltaRow <= 1; ++deltaRow) {
		int neighbourRow = row + deltaRow;
		if (neighbourRow < 0 || neighbourRow >= g_height) continue;

		for (int deltaColumn = -1; deltaColumn <= 1; ++deltaColumn) {
			if (deltaRow == 0 && deltaColumn == 0) continue;
			int neighbourColumn = column + deltaColumn;
			if (neighbourColumn < 0 || neighbourColumn >= g_width) continue;

			int neighbour = neighbourRow * g_width + neighbourColumn;
			if (g_properties[neighbour].valid == 0u) continue;
			if (abs(g_vertices[neighbour].z - depth) > tolerance) continue;
			count += 1;
		}
	}
	return count;
}

void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int centre = row * g_width + column;
	g_normals[centre] = vec4(0.0);
	// Only `emitted` is written here. The forward guard's verdict must NOT reach `valid`: a pixel
	// it refuses still HAS a measurement, and [H4]/[H5] count neighbours by `valid`, so folding
	// the two together would shift every neighbour count downstream.
	g_properties[centre].emitted = 0u;

	// BackprojectDepth walks [0,W-1) x [0,H-1) because the forward difference needs u+1 and v+1.
	// The last row and column therefore never emit, and matching that exactly is what lets the
	// GPU and CPU point lists be compared index by index.
	if (column + 1 >= g_width || row + 1 >= g_height) return;

	int right = centre + 1;
	int below = centre + g_width;
	if (g_properties[centre].valid == 0u || g_properties[right].valid == 0u ||
	    g_properties[below].valid == 0u)
		return;

	// A step in depth is two surfaces, not one: differencing across it yields a normal belonging
	// to neither -- the flying pixel a stereo sensor produces at every object boundary.
	float depth   = g_vertices[centre].z;
	float maxJump = DepthJumpTolerance(depth, g_relativeDepthJump, g_minimumDepthJump);
	if (abs(g_vertices[right].z - depth) > maxJump) return;
	if (abs(g_vertices[below].z - depth) > maxJump) return;

	// [H4] The forward test above protects the NORMAL, which is differenced from exactly those two
	// neighbours. Whether the POINT is trustworthy is a different question, answered on the
	// trailing edge of the step the forward test cannot see.
	if (g_symmetricDepthJumpGuard != 0u && StraddlesADepthStep(column, row, depth, maxJump)) return;

	// [H5] Those two neighbours agreeing is just as true of a mismatched island as of real
	// surface. How much of the neighbourhood exists at all is what separates them.
	if (g_minimumValidNeighbours > 0 &&
	    CountSameSurfaceNeighbours(column, row, depth, maxJump) < g_minimumValidNeighbours)
	{
		atomicAdd(g_counters.rejectedByNeighbourSupport, 1u);
		return;
	}

	vec3 point  = g_vertices[centre].xyz;
	vec3 normal = cross(g_vertices[right].xyz - point, g_vertices[below].xyz - point);
	if (length(normal) < 1e-9) return;
	normal = normalize(normal);

	// The camera sits at the frame origin, so `point` IS the view direction. Testing normal.z
	// alone agrees only on the optical axis; across a wide field of view an ordinary wall
	// receding toward the image edge inverts, and an inverted normal flips the sign of the TSDF
	// update and of every point-to-plane residual.
	if (dot(normal, point) > 0.0) normal = -normal;

	// [H6] Incidence against the pixel's OWN ray, for the same reason the orientation flip above
	// uses it. `normal` is camera-facing here, so -dot(normal, rayDirection) is cos(incidence).
	if (g_minimumIncidenceCosine > 0.0 &&
	    -dot(normal, normalize(point)) < g_minimumIncidenceCosine)
	{
		atomicAdd(g_counters.rejectedByIncidence, 1u);
		return;
	}

	g_normals[centre] = vec4(normal, 0.0);
	g_properties[centre].emitted = 1u;
	atomicAdd(g_counters.emittedPoints, 1u);
}
