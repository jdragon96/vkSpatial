#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Mirrored by ValidationMaskProperty in ValidationMask.h. 4-byte scalars only.
struct ValidationMaskProperty {
	uint valid;
};

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	float g_fx;
	float g_fy;
	float g_cx;
	float g_cy;
	float g_minimumDepthMeters;   // 0 disables this end
	float g_maximumDepthMeters;   // 0 disables this end
};

layout(std430, set = 0, binding = 0) readonly  buffer SourceDepth { float g_depth[]; };
// vec4, not vec3: std430 gives a vec3 array a 16-byte stride anyway, so the padding is spelled out
// rather than left for a CPU mirror to get wrong. w is unused.
layout(std430, set = 0, binding = 1) writeonly buffer VertexGrid  { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 2) writeonly buffer ValidMask   { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 3) buffer Counters { uint g_rejectedByRange; };

bool InRange(float depth)
{
	if (g_minimumDepthMeters > 0.0 && depth < g_minimumDepthMeters) return false;
	if (g_maximumDepthMeters > 0.0 && depth > g_maximumDepthMeters) return false;
	return true;
}

void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int   centre = row * g_width + column;
	float depth  = g_depth[centre];

	// A dropout and a range rejection are both invalid but are NOT the same event: one is the
	// sensor refusing to match, the other is a threshold the caller chose. Only the second is
	// counted, or the counter stops being usable for tuning the threshold.
	bool present = depth > 0.0;
	bool admitted = present && InRange(depth);
	if (present && !admitted) atomicAdd(g_rejectedByRange, 1u);

	if (!admitted)
	{
		// Zeroed, not left stale: a rejected pixel must not be reachable by reading the grid
		// without also reading the mask.
		g_vertices[centre] = vec4(0.0);
		g_properties[centre].valid = 0u;
		return;
	}

	g_vertices[centre] = vec4((float(column) - g_cx) / g_fx * depth,
	                          (float(row) - g_cy) / g_fy * depth,
	                          depth,
	                          0.0);
	g_properties[centre].valid = 1u;
}
