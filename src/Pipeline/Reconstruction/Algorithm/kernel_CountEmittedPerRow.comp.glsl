#version 450
#include "ValidationMask.common.glsl"

layout(local_size_x = 64) in;

layout(push_constant) uniform PC
{
	int g_width;
	int g_height;
};

layout(std430, set = 0, binding = 0) readonly  buffer ValidMask { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 1) writeonly buffer RowCount  { uint g_rowCount[]; };

void main()
{
	int row = int(gl_GlobalInvocationID.x);
	if (row >= g_height) return;

	uint count = 0u;
	for (int column = 0; column < g_width; ++column)
		count += g_properties[row * g_width + column].emitted;
	g_rowCount[row] = count;
}
