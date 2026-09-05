/// *********************************************
/// "central" -- symmetric differences over the same 3x3 footprint.
///
///     n = ((P(u+1,v) - P(u-1,v)) / 2) x ((P(u,v+1) - P(u,v-1)) / 2)
///
/// Two changes against "forward", both free:
///
///  - The estimate is anchored ON the pixel instead of at the corner of a triangle, so the
///    half-pixel shift of the normal field disappears.
///  - The tangent spans two pixels rather than one, so a difference of two samples with noise
///    sigma is divided by 2 instead of 1: the gradient carries sigma/sqrt(2) where "forward"
///    carries sqrt(2)*sigma. Half the angular noise, for the same four loads.
///
/// It does NOT fall back to a one-sided difference when a partner is missing. A frame whose normals
/// came from two different estimators cannot answer "did the symmetric stencil help", which is the
/// only reason to have this axis; the pixel is refused instead and charged to the stencil counter.
/// *********************************************

int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
{
	normal = vec3(0.0);

	// 1. The stencil reaches one pixel either side, so a one-pixel border has no domain.
	if (column - 1 < 0 || column + 1 >= g_width || row - 1 < 0 || row + 1 >= g_height)
		return NORMAL_ESTIMATE_OUT_OF_DOMAIN;

	// 2. All four partners must be on this pixel's surface.
	vec3 left, right, above, below;
	if (!SameSurfaceSample(column - 1, row, depth, tolerance, left)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column + 1, row, depth, tolerance, right)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row - 1, depth, tolerance, above)) return NORMAL_ESTIMATE_NO_SUPPORT;
	if (!SameSurfaceSample(column, row + 1, depth, tolerance, below)) return NORMAL_ESTIMATE_NO_SUPPORT;

	// 3. Cross the two central differences. The 0.5 factors scale the normal, not its direction,
	//    and normalize() drops them -- they are written out because the tangents are what the
	//    noise argument above is about, not the cross product.
	vec3 tangentAlongColumns = (right - left) * 0.5;
	vec3 tangentAlongRows    = (below - above) * 0.5;

	normal = cross(tangentAlongColumns, tangentAlongRows);
	if (length(normal) < 1e-9) return NORMAL_ESTIMATE_NO_SUPPORT;

	normal = normalize(normal);
	return NORMAL_ESTIMATE_OK;
}
