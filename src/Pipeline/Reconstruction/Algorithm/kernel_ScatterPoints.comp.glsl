#version 450
#include "kernel_ValidationMaskCommmon.glsl"

layout(local_size_x = 64) in;

layout(push_constant) uniform PC
{
	int g_width;
	int g_height;
};

layout(std430, set = 0, binding = 0) readonly  buffer ValidMask     { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 1) readonly  buffer VertexGrid    { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 2) readonly  buffer NormalGrid    { vec4 g_normals[]; };
layout(std430, set = 0, binding = 3) readonly  buffer RowOffset     { uint g_rowOffset[]; };
layout(std430, set = 0, binding = 4) writeonly buffer CompactPoints { vec4 g_points[]; };
layout(std430, set = 0, binding = 5) writeonly buffer CompactNormals{ vec4 g_compactNormals[]; };

void main()
{
	int row = int(gl_GlobalInvocationID.x);
	if (row >= g_height) return;

	// One invocation owns a whole row and walks it in order, so the compacted array comes out in
	// exactly the row-major order BackprojectDepth emits in.
	uint slot = g_rowOffset[row];
	for (int column = 0; column < g_width; ++column)
	{
		int pixel = row * g_width + column;
		if (g_properties[pixel].emitted == 0u) continue;
		g_points[slot]         = g_vertices[pixel];
		g_compactNormals[slot] = g_normals[pixel];
		slot += 1u;
	}
}
