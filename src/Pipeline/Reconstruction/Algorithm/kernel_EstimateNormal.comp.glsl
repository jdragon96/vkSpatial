#version 450
#include "kernel_ValidationMaskCommmon.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	float g_relativeDepthJump;
	float g_minimumDepthJump;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexGrid { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 1) buffer ValidMask { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2) writeonly buffer NormalGrid { vec4 g_normals[]; };

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

	vec3 point  = g_vertices[centre].xyz;
	vec3 normal = cross(g_vertices[right].xyz - point, g_vertices[below].xyz - point);
	if (length(normal) < 1e-9) return;
	normal = normalize(normal);

	// The camera sits at the frame origin, so `point` IS the view direction. Testing normal.z
	// alone agrees only on the optical axis; across a wide field of view an ordinary wall
	// receding toward the image edge inverts, and an inverted normal flips the sign of the TSDF
	// update and of every point-to-plane residual.
	if (dot(normal, point) > 0.0) normal = -normal;

	g_normals[centre] = vec4(normal, 0.0);
	g_properties[centre].emitted = 1u;
}
