#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;
	int   g_window;              // side of the square window in pixels; <= 1 = passthrough
	float g_relativeDepthJump;   // fraction of range
	float g_minimumDepthJump;    // metres, near-field floor
};

// Row-major width * height, metres. <= 0 means the matcher produced no depth for that pixel.
layout(std430, set = 0, binding = 0) readonly buffer SourceDepth   { float g_source[]; };
layout(std430, set = 0, binding = 1) writeonly buffer FilteredDepth { float g_filtered[]; };

float DepthJumpTolerance(float depth)
{
	return max(g_minimumDepthJump, g_relativeDepthJump * depth);
}


void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int centre = row * g_width + column;
	float centreDepth = g_source[centre];

	if (g_window <= 1)
	{
		g_filtered[centre] = centreDepth;
		return;
	}

	if (centreDepth <= 0.0)
	{
		g_filtered[centre] = 0.0;
		return;
	}

	float tolerance = DepthJumpTolerance(centreDepth);
	int   radius    = g_window / 2;
	float sum       = 0.0;
	int   count     = 0;

	for (int deltaRow = -radius; deltaRow <= radius; ++deltaRow) {
		int neighbourRow = row + deltaRow;
		if (neighbourRow < 0 || neighbourRow >= g_height) continue;

		for (int deltaColumn = -radius; deltaColumn <= radius; ++deltaColumn) {
			int neighbourColumn = column + deltaColumn;
			if (neighbourColumn < 0 || neighbourColumn >= g_width) continue;

			float neighbourDepth = g_source[neighbourRow * g_width + neighbourColumn];
			if (neighbourDepth <= 0.0) continue;
			if (abs(neighbourDepth - centreDepth) > tolerance) continue;

			sum += neighbourDepth;
			count += 1;
		}
	}

	g_filtered[centre] = (count > 0) ? (sum / float(count)) : centreDepth;
}
