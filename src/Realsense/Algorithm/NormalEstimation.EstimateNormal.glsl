#version 450

#include "Realsense/Algorithm/Common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC
{
	int   g_width;
	int   g_height;

	float g_subpixelRms;
	float g_focalLengthPixels;
	float g_baselineMeters;
	float g_sameSurfaceSigmaMultiplier;
	float g_depthScale;

	int   g_planeFitRadius;
	int   g_minimumPlaneFitSamples;
};

layout(std430, set = 0, binding = 0) readonly  buffer VertexGrid { vec4 g_vertices[]; };
layout(std430, set = 0, binding = 1)           buffer ValidMask  { ValidationMaskProperty g_properties[]; };
layout(std430, set = 0, binding = 2) writeonly buffer NormalGrid { vec4 g_normals[]; };
layout(std430, set = 0, binding = 3)           buffer Counters   { NormalEstimationCounters g_counters; };

#include "Realsense/Algorithm/NormalEstimation.Strategy.glsl"

/// The normal pass may only CANCEL a point, never elect one.
///
/// Which pixels are wanted was decided by the score threshold in ValidationMask.RemainValidDepth;
/// re-electing here would let a pixel the confidence rejected back in through a second door. What
/// this pass adds is a veto: a point without a normal is useless to point-to-plane ICP and to a
/// weighted TSDF fusion alike, so `emitted == 1` is made to mean "coordinate AND normal" and no
/// consumer downstream has to special-case a zero normal.
void main()
{
	int column = int(gl_GlobalInvocationID.x);
	int row    = int(gl_GlobalInvocationID.y);
	if (column >= g_width || row >= g_height) return;

	int centre = row * g_width + column;
	g_normals[centre] = vec4(0.0);

	// 1. Nothing to do where the threshold pass already dropped the pixel. Reading `emitted`
	//    rather than `valid` keeps this pass off the pixels nobody asked for.
	if (g_properties[centre].emitted == 0u) return;

	float depth     = g_vertices[centre].z;
	float tolerance = SameSurfaceTolerance(depth, g_subpixelRms, g_focalLengthPixels,
	                                       g_baselineMeters, g_sameSurfaceSigmaMultiplier,
	                                       g_depthScale);

	// 2. Estimate. The fragment compiled in decides HOW; the two failure causes are charged here
	//    so the attribution does not depend on which fragment it was.
	vec3 normal;
	int outcome = EstimateSurfaceNormal(column, row, depth, tolerance, normal);
	if (outcome == NORMAL_ESTIMATE_OUT_OF_DOMAIN)
	{
		g_properties[centre].emitted = 0u;
		atomicAdd(g_counters.outOfDomain, 1u);
		return;
	}
	if (outcome == NORMAL_ESTIMATE_NO_SUPPORT)
	{
		g_properties[centre].emitted = 0u;
		atomicAdd(g_counters.noSupport, 1u);
		return;
	}

	// 3. Orient against this pixel's own view ray. In camera space the ray to the point IS the
	//    point, so the sign of the dot product is the whole test.
	vec3 point = g_vertices[centre].xyz;
	if (dot(normal, point) > 0.0) normal = -normal;

	g_normals[centre] = vec4(normal, 0.0);
}
