#version 450

///////////////////////////////////////////////////////////////////////////////////////////////
// [H1] PrefilterDepth -- discontinuity-aware square mean over a depth image.
//
// GPU port of Pipeline::PrefilterDepth (DepthCameraFrameSource.h). Same output, one invocation
// per pixel. See docs/DEPTH_NOISE_FILTERING.md section [H1].
//
// The normal estimated downstream is a ONE-PIXEL forward difference, so its conditioning is set
// entirely by per-pixel depth noise. Averaging only the neighbours that lie on the SAME surface
// is what makes this a denoiser rather than a blur: a plain box mean would average across every
// object boundary and manufacture exactly the flying pixels the later guards exist to remove.
//
// Source and destination MUST be distinct buffers. Beyond the obvious read/write race, an
// in-place filter would feed already-smoothed values back into later windows, so the result would
// depend on invocation order -- and this repo's replay determinism depends on it not doing that.
///////////////////////////////////////////////////////////////////////////////////////////////

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

/// *********************************************
/// Tolerance
/// *********************************************

/// Depth-jump tolerance at range `depth`. Shared with [H3] [H4] [H5] downstream; the four must
/// agree or "same surface" means something different in each.
///
/// Relative because stereo depth error grows as z^2/(f*B) (Keselman et al. eq. 2): one fixed
/// threshold over-rejects near the camera and under-rejects far from it. The absolute floor keeps
/// it from collapsing to nothing in the near field.
float DepthJumpTolerance(float depth)
{
	return max(g_minimumDepthJump, g_relativeDepthJump * depth);
}

/// *********************************************
/// Filter
/// *********************************************

void main()
{
	// 1. One invocation per pixel. The dispatch is rounded up to the workgroup size, so the
	//    surplus invocations must leave the buffer alone rather than wrap onto row 0.
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int centre = row * g_width + column;
	float centreDepth = g_source[centre];

	// 2. Passthrough cases. Both still WRITE: the destination is a separate buffer and nothing
	//    else fills it, so an early return without a store would leave it undefined.
	// 2.1. A window of one pixel is the identity, by definition.
	if (g_window <= 1)
	{
		g_filtered[centre] = centreDepth;
		return;
	}
	// 2.2. Invalid stays invalid. Smoothing a pixel the matcher refused would invent surface
	//      out of its neighbours, which is the opposite of what this pass is for.
	if (centreDepth <= 0.0)
	{
		g_filtered[centre] = 0.0;
		return;
	}

	// 3. Accumulate the neighbours that are valid AND on the centre's own surface.
	//    The tolerance comes from the CENTRE's depth, not from each neighbour's: the question
	//    being asked is "does this neighbour belong to the surface under this pixel".
	float tolerance = DepthJumpTolerance(centreDepth);
	int   radius    = g_window / 2;
	float sum       = 0.0;
	int   count     = 0;

	for (int deltaRow = -radius; deltaRow <= radius; ++deltaRow) {
		int neighbourRow = row + deltaRow;
		// 3.1. Clip the window at the image border rather than discarding the pixel. Dropping
		//      border pixels would silently crop every frame by `radius` on all four sides.
		if (neighbourRow < 0 || neighbourRow >= g_height) continue;

		for (int deltaColumn = -radius; deltaColumn <= radius; ++deltaColumn) {
			int neighbourColumn = column + deltaColumn;
			if (neighbourColumn < 0 || neighbourColumn >= g_width) continue;

			// 3.2. Two rejections, different reasons: no measurement at all, or a measurement
			//      belonging to another surface.
			float neighbourDepth = g_source[neighbourRow * g_width + neighbourColumn];
			if (neighbourDepth <= 0.0) continue;
			if (abs(neighbourDepth - centreDepth) > tolerance) continue;

			sum += neighbourDepth;
			count += 1;
		}
	}

	// 4. The centre is inside its own window and trivially within tolerance of itself, so count
	//    is always >= 1 here. The guard is kept because a zero divisor would not fail loudly on
	//    the GPU -- it would write inf into the depth image and corrupt everything downstream.
	g_filtered[centre] = (count > 0) ? (sum / float(count)) : centreDepth;
}
