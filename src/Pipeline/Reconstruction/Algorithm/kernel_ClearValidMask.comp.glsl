#version 450

layout(local_size_x = 16, local_size_y = 16) in;

struct ValidationMaskProperty {
	uint valid;
	uint emitted;   // 0 = no measurement
};

layout(std430, set = 0, binding = 0) writeonly buffer ValidMask { ValidationMaskProperty g_properties[]; };

layout(push_constant) uniform PC
{
	int g_width;
	int g_height;
};

void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	g_properties[row * g_width + column].valid = 0u;
	g_properties[row * g_width + column].emitted = 0u;
}
