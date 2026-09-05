/// *********************************************
/// "forward" -- one forward-difference triangle. The historical estimator, kept as the noisy
/// baseline every other strategy is measured against.
///
///     n = (P(u+1,v) - P(u,v)) x (P(u,v+1) - P(u,v))
///
/// Two samples per tangent, so each tangent carries sqrt(2) * sigma of the per-pixel depth noise.
/// It is also anchored at a corner: the normal of that triangle is the surface normal at
/// (u+0.5, v+0.5), stored at (u,v), which shifts the whole normal field half a pixel on a curved
/// surface. "central" fixes both at the same cost.
/// *********************************************

int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
{
	normal = vec3(0.0);

	// 1. The stencil reaches u+1 and v+1, so the last row and column have no domain.
	if (column + 1 >= g_width || row + 1 >= g_height) return NORMAL_ESTIMATE_OUT_OF_DOMAIN;

	// 2. Both differencing partners must be on this pixel's surface. A step in depth is two
	//    surfaces, not one, and a normal differenced across it belongs to neither.
	vec3 right, below;
	if (!SameSurfaceSample(column + 1, row, depth, tolerance, right)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row + 1, depth, tolerance, below)) return NORMAL_ESTIMATE_NO_SUPPORT;

	// 3. Cross the two edges of the triangle.
	vec3 point = g_vertices[row * g_width + column].xyz;
	normal = cross(right - point, below - point);
	if (length(normal) < 1e-9) return NORMAL_ESTIMATE_NO_SUPPORT;

	normal = normalize(normal);
	return NORMAL_ESTIMATE_OK;
}
