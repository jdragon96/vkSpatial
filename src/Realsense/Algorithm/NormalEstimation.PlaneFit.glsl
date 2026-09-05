/// *********************************************
/// "planefit" -- total-least-squares plane through every same-surface sample in a k x k window.
///
/// The normal is the eigenvector of the smallest eigenvalue of the samples' covariance: the
/// direction of least spread, which for points on a surface patch is the surface normal.
///
/// Why this rather than a wider difference: the slope of a least-squares fit over k evenly spaced
/// samples carries sigma * sqrt(12 / (k(k^2 - 1))) of the per-sample noise -- 0.32*sigma at k = 5
/// against 1.41*sigma for a forward difference. That is 4.5x less gradient noise, and it is bought
/// by using MORE MEASUREMENTS rather than by smoothing the depth image first, so the point this
/// normal belongs to stays exactly where the sensor put it.
///
/// Unlike a difference stencil the fit degrades gracefully: it uses whatever same-surface samples
/// the window holds and refuses only below g_minimumPlaneFitSamples, so a hole in the window costs
/// accuracy rather than the whole pixel.
/// *********************************************

/// Smallest eigenvalue of a symmetric 3x3, by the closed-form trigonometric solution of its
/// characteristic cubic (the roots of a real symmetric matrix are real, so no iteration is needed
/// and no branch diverges across the workgroup).
///
/// `m` is expected TRACE-NORMALISED, so its eigenvalues sum to 1 and the epsilons below are
/// absolute rather than scale-dependent -- see the caller for why that matters here.
float SmallestEigenvalueOfSymmetric(mat3 m)
{
	// GLSL indexes a mat3 column-first, but m is symmetric, so m[i] is equally its i-th row.
	float offDiagonalNorm = m[1][0] * m[1][0] + m[2][0] * m[2][0] + m[2][1] * m[2][1];
	float mean = (m[0][0] + m[1][1] + m[2][2]) / 3.0;

	// 1. Already diagonal: the eigenvalues are the diagonal itself.
	if (offDiagonalNorm <= 1e-18) return min(m[0][0], min(m[1][1], m[2][2]));

	float spread = (m[0][0] - mean) * (m[0][0] - mean) +
	               (m[1][1] - mean) * (m[1][1] - mean) +
	               (m[2][2] - mean) * (m[2][2] - mean) + 2.0 * offDiagonalNorm;

	// 2. Isotropic: three equal eigenvalues, so the "least spread" direction does not exist. The
	//    caller sees it as a null-space of dimension 3 and refuses the pixel.
	if (spread <= 1e-18) return mean;

	float radius = sqrt(spread / 6.0);
	mat3 normalised = (1.0 / radius) * (m - mean * mat3(1.0));

	// 3. det/2 of the normalised matrix is cos(3*phi) for the angle that separates the three roots.
	//    The clamp is against rounding pushing it a hair outside acos()'s domain.
	float angle = acos(clamp(determinant(normalised) * 0.5, -1.0, 1.0)) / 3.0;

	// 4. The three roots sit 2*pi/3 apart on that circle; the +2*pi/3 branch is the smallest.
	return mean + 2.0 * radius * cos(angle + 2.0943951023931953);
}

/// A vector spanning the null space of `m`, which the caller passes as (covariance - lambda*I) for
/// the smallest eigenvalue -- a rank-2 matrix whose null direction is the eigenvector.
///
/// Two rows of a rank-2 symmetric matrix span its row space, so their cross product is the null
/// direction. All three pairs are tried and the longest kept: any single pair can be nearly
/// parallel, and its cross product would then be dominated by rounding rather than by the surface.
vec3 NullDirectionOfSymmetric(mat3 m)
{
	vec3 candidates[3];
	candidates[0] = cross(m[0], m[1]);
	candidates[1] = cross(m[0], m[2]);
	candidates[2] = cross(m[1], m[2]);

	vec3 best = candidates[0];
	float bestLengthSquared = dot(candidates[0], candidates[0]);
	for (int i = 1; i < 3; ++i)
	{
		float lengthSquared = dot(candidates[i], candidates[i]);
		if (lengthSquared > bestLengthSquared)
		{
			best = candidates[i];
			bestLengthSquared = lengthSquared;
		}
	}
	return best;
}

int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
{
	normal = vec3(0.0);

	int radius = g_planeFitRadius;
	if (column - radius < 0 || column + radius >= g_width ||
	    row - radius < 0 || row + radius >= g_height)
		return NORMAL_ESTIMATE_OUT_OF_DOMAIN;

	// 1. Accumulate the moments of the window's same-surface samples, measured FROM THE CENTRE
	//    POINT rather than from the origin.
	//
	//    This is not cosmetic. The covariance of a surface patch is the difference of two nearly
	//    equal quantities: at 1.5 m range the coordinates are O(1 m) while the spread across the
	//    window is O(1 mm), so E[p^2] - E[p]^2 in float32 would cancel about seven digits and leave
	//    roughly one -- the eigenvector would be rounding noise. Shifting the origin to the centre
	//    point makes every accumulated value O(mm) and the cancellation harmless.
	vec3 centrePoint = g_vertices[row * g_width + column].xyz;
	vec3 offsetSum = vec3(0.0);
	vec3 squareSum = vec3(0.0);      // (xx, yy, zz)
	vec3 productSum = vec3(0.0);     // (xy, xz, yz)
	int sampleCount = 0;

	for (int deltaRow = -radius; deltaRow <= radius; ++deltaRow)
	{
		for (int deltaColumn = -radius; deltaColumn <= radius; ++deltaColumn)
		{
			vec3 neighbourPoint;
			if (!SameSurfaceSample(column + deltaColumn, row + deltaRow, depth, tolerance, neighbourPoint))
				continue;

			vec3 offset = neighbourPoint - centrePoint;
			offsetSum += offset;
			squareSum += offset * offset;
			productSum += vec3(offset.x * offset.y, offset.x * offset.z, offset.y * offset.z);
			sampleCount += 1;
		}
	}

	// 2. Three non-collinear samples are the algebraic minimum; the caller's threshold is what
	//    decides how over-determined the fit has to be.
	if (sampleCount < max(3, g_minimumPlaneFitSamples)) return NORMAL_ESTIMATE_NO_SUPPORT;

	float inverseCount = 1.0 / float(sampleCount);
	vec3 offsetMean = offsetSum * inverseCount;
	vec3 variance = squareSum * inverseCount - offsetMean * offsetMean;
	vec3 covariance = productSum * inverseCount - vec3(offsetMean.x * offsetMean.y,
	                                                   offsetMean.x * offsetMean.z,
	                                                   offsetMean.y * offsetMean.z);

	// 3. Normalise by the trace before solving. The entries are O(mm^2) = O(1e-6), so their cubes
	//    -- which is what a determinant and a null-space cross product are made of -- land near
	//    float32's smallest normals, and any absolute epsilon would reject every real surface.
	//    Scaling a matrix scales its eigenvalues and leaves its eigenvectors untouched.
	float trace = variance.x + variance.y + variance.z;
	if (trace <= 0.0) return NORMAL_ESTIMATE_NO_SUPPORT; // every sample landed on one point

	mat3 scatter = mat3(variance.x,   covariance.x, covariance.y,
	                    covariance.x, variance.y,   covariance.z,
	                    covariance.y, covariance.z, variance.z) * (1.0 / trace);

	float smallest = SmallestEigenvalueOfSymmetric(scatter);
	normal = NullDirectionOfSymmetric(scatter - smallest * mat3(1.0));

	// 4. A null space wider than one dimension means the samples never pinned a plane down --
	//    collinear ones (a one-pixel-wide strip of surface) leave two directions of least spread,
	//    and every pairwise cross product collapses. With the trace normalised above, a genuine
	//    plane leaves this length near 0.25, so the threshold is nowhere near a real surface.
	if (dot(normal, normal) < 1e-12) return NORMAL_ESTIMATE_NO_SUPPORT;

	normal = normalize(normal);
	return NORMAL_ESTIMATE_OK;
}
