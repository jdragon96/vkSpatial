#version 450

#include "Realsense/Algorithm/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	float g_fx;
	float g_fy;
	float g_cx;
	float g_cy;
	float g_depthScale;
	float g_scoreThreshold;
};

layout(set = 0, binding = 0) uniform usampler2D g_depth;                         // R16_UINT, Z16
layout(std430, set = 0, binding = 1) buffer          ValidMask  { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2) writeonly buffer VertexGrid { vec4 g_vertices[]; };

vec3 BackProject(int column, int row, float depth)
{
	return vec3((float(column) - g_cx) / g_fx * depth,
	            (float(row) - g_cy) / g_fy * depth,
	            depth);
}

void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int pixel = row * g_width + column;
	g_vertices[pixel] = vec4(0.0);
	g_properties[pixel].emitted = 0u;

	if (g_properties[pixel].valid == 0u) return;

	float depth = float(texelFetch(g_depth, ivec2(column, row), 0).x) * g_depthScale;
	if (depth <= 0.0) return;

	g_vertices[pixel] = vec4(BackProject(column, row, depth), 0.0);
	if (g_properties[pixel].score >= g_scoreThreshold) g_properties[pixel].emitted = 1u;
}
