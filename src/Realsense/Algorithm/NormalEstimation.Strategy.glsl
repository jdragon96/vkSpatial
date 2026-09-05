/// *********************************************
/// Normal estimator dispatcher
///
/// NormalEstimation.EstimateNormal.glsl includes only this file. Which fragment it pulls in is
/// decided by a preprocessor definition the C++ side passes through ComputePipeline::Define,
/// listed in Realsense/Algorithm/NormalEstimation.h.
///
/// Why a dispatcher instead of `#include NORMAL_STRATEGY_INCLUDE`: glslang does NOT macro-expand
/// #include ("must be followed by a header name"), so the include name has to be a literal. This
/// keeps each fragment a real file that editors and grep can follow.
///
/// The fragment includes below use the full src-relative path (not a bare filename):
/// ComputePipeline resolves an #include against a fixed root list computed once for the top-level
/// compiled file -- [that file's own folder, VKBVH_SHADER_DIR, VKBVH_SRC_DIR] -- not against THIS
/// file's folder, even though this file is itself reached via #include.
///
/// Every fragment defines exactly one entry point:
///
///     int EstimateSurfaceNormal(int column, int row, float depth, float tolerance, out vec3 normal)
///
/// It returns an UNORIENTED unit normal. Orientation against the pixel's own view ray, the
/// `emitted` veto and both counters stay in the kernel: which cause a rejection is charged to must
/// not depend on which fragment happened to be compiled in.
///
/// The fragments read g_vertices, g_properties, g_width and g_height, which the including kernel
/// declares above this include.
/// *********************************************

/// Outcome of one estimator call.
///
/// The two failures are kept apart because only one of them is a discarded measurement. A stencil
/// hanging off the edge of the image is the estimator's domain ending -- every estimator loses a
/// border, and a wider one loses more -- while a stencil that fits and still cannot find
/// same-surface samples is a pixel the scene refused, which is what the counter is for.
#define NORMAL_ESTIMATE_OK            0
#define NORMAL_ESTIMATE_OUT_OF_DOMAIN 1
#define NORMAL_ESTIMATE_NO_SUPPORT    2

/// A neighbour is usable only if it exists, carries a measurement, and sits on the centre pixel's
/// surface. Every fragment goes through this one definition so "same surface" cannot come to mean
/// three different things across the three estimators -- the same reason SameSurfaceTolerance in
/// Common.glsl is shared with the score kernel.
bool SameSurfaceSample(int column, int row, float depth, float tolerance, out vec3 point)
{
	point = vec3(0.0);
	if (column < 0 || column >= g_width || row < 0 || row >= g_height) return false;

	int index = row * g_width + column;
	if (g_properties[index].valid == 0u) return false;
	if (abs(g_vertices[index].z - depth) > tolerance) return false;

	point = g_vertices[index].xyz;
	return true;
}

#if defined(NORMAL_PLANE_FIT)
#include "Realsense/Algorithm/NormalEstimation.PlaneFit.glsl"
#elif defined(NORMAL_CENTRAL_DIFFERENCE)
#include "Realsense/Algorithm/NormalEstimation.CentralDifference.glsl"
#else
#include "Realsense/Algorithm/NormalEstimation.ForwardDifference.glsl"
#endif
