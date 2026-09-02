#version 460

/// *********************************************
/// Constants
/// *********************************************
layout(local_size_x = 256) in;

#define TSDF_SCALE           10000.0
#define MIN_RELATIVE_WEIGHT  0.05   // drop directions weaker than 5% of the dominant one

#include "voxel_common.glsl"   // EMPTY_KEY, MAX_PROBE, wangHash

// 24 bytes: distance/weight accumulators + stored-gradient (observed normal) accumulators.
// Layout must match DirEntry in kernel_AdvancedTSDF.extract.comp.glsl and AdvancedTSDF.h.
struct DirEntry
{
	uint key;
	int  sumDW;   // Σ tsdf · w · TSDF_SCALE
	uint sumW;    // Σ w · TSDF_SCALE
	int  sumNx;   // Σ n · w · TSDF_SCALE  (stored gradient; normalized at extraction)
	int  sumNy;
	int  sumNz;
};

layout(push_constant) uniform PC {
	uint  g_numPoints;
	uint  g_hashCapacity;
	float g_voxelSize;
	float g_truncation;
	float g_camX;
	float g_camY;
	float g_camZ;
	uint  g_maxDirections;
	uint  g_dirExponent;
	uint  g_viewAngleWeight;
	int   g_originX;
	int   g_originY;
	int   g_originZ;
	uint  g_pointToPlane;   // 1 = point-to-plane SDF (removes grazing bias), 0 = projective
	float g_confWeight;     // A1: surface-proximity confidence lambda in [0,1] (0 = uniform/off)
	int   g_currentFrame;   // stamped into g_firstFrame[slot] the first time a slot is filled

	// Range-adaptive truncation band (see AdvancedTSDF::AdaptiveBandConfig).
	// g_bandSigmaMultiplier == 0 disables it and the band is g_truncation everywhere.
	float g_bandSigmaMultiplier;
	float g_bandMinimumVoxels;
	float g_sigmaConstant;
	float g_sigmaQuadratic;
	float g_sigmaOffsetMeters;
	float g_sigmaAngular;

	// 1 = discount only the occluded side of the band (Bylow / Voxblox eq. 5);
	// 0 = the symmetric A1 profile driven by g_confWeight.
	uint  g_behindSurfaceDropoff;
};

layout(std430, set = 0, binding = 0) buffer HashTable
{
	DirEntry g_hash[];
};
layout(std430, set = 0, binding = 1) readonly buffer Points
{
	float g_points[];
};
layout(std430, set = 0, binding = 2) readonly buffer Normals
{
	float g_normals[];
};
layout(set = 0, binding = 3) buffer Stat
{
	uint g_filledCount;
};
// Per-slot "first frame that filled it" -- stamped once on insert, read back at compaction so the CPU
// never has to re-hash the whole model to find newly-filled voxels (replaces the CPU first-seen tracker).
layout(std430, set = 0, binding = 4) buffer FirstFrame
{
	int g_firstFrame[];
};
// Observations dropped because probing gave up. Must stay 0 in a healthy run -- a non-zero value
// means the table's load factor limit is set too high for this hash strategy.
layout(std430, set = 0, binding = 5) buffer InsertFailures
{
	uint g_insertFailureCount;
};

/// *********************************************
/// Hashing
/// *********************************************
bool packDirKey(ivec3 voxel, uint direction, out uint key)
{
	ivec3 localVoxel = voxel - ivec3(g_originX, g_originY, g_originZ);
	if (localVoxel.x < 0 || localVoxel.x > 511 ||
		localVoxel.y < 0 || localVoxel.y > 511 ||
		localVoxel.z < 0 || localVoxel.z > 511)
	{
		key = 0u;
		return false;
	}
	key = ((uint(localVoxel.x) & 0x1FFu) << 21u)
		| ((uint(localVoxel.y) & 0x1FFu) << 12u)
		| ((uint(localVoxel.z) & 0x1FFu) <<  3u)
		|  (direction & 0x7u);
	return true;
}

#define HASH_WITH_INSERT
#include "TSDF/Hash/HashStrategy.glsl"

/// *********************************************
/// Direction weighting
/// *********************************************
float ipow(float base, uint exponent)
{
	float result = 1.0;
	for (uint i = 0u; i < exponent; i++) result *= base;
	return result;
}

uint signedAxisIndex(uint axis, float n_c)
{
	return 2u * axis + (n_c >= 0.0 ? 0u : 1u);
}

int selectDirections(
	vec3 normal,
	out uint descDirection[3],
	out float reliability[3])
{
	float align[3] = float[](
		abs(normal.x),
		abs(normal.y),
		abs(normal.z) );
	uint axis[3] = uint[](
		signedAxisIndex(0u, normal.x),
		signedAxisIndex(1u, normal.y),
		signedAxisIndex(2u, normal.z)
	);

	// Descending insertion sort by alignment.
	for (int i = 1; i < 3; i++) {
		float a = align[i]; 
		uint ax = axis[i];
		int j = i - 1;
		while (j >= 0 && align[j] < a)
		{
			align[j + 1] = align[j];
			axis[j + 1] = axis[j];
			j--;
		}
		align[j + 1] = a; 
		axis[j + 1] = ax;
	}

	// The strongest axis is always kept, at relative weight 1.
	descDirection[0] = axis[0];
	reliability[0] = 1.0;
	int count = 1;
	float dominantWeight = ipow(align[0], g_dirExponent);
	if (dominantWeight <= 0.0) return count;
	uint maxDirs = clamp(g_maxDirections, 1u, 3u);

	for (int i = 1; i < 3 && uint(count) < maxDirs; i++) {
		float relativeWeight = ipow(align[i], g_dirExponent);
		relativeWeight /= dominantWeight;
		if (relativeWeight < MIN_RELATIVE_WEIGHT) continue;
		descDirection[count] = axis[i];
		reliability[count] = relativeWeight;
		count++;
	}

	return count;
}

/// *********************************************
/// Adaptive band
/// *********************************************

/// Empirical axial depth noise, Nguyen/Izadi/Lovell 3DIMPVT 2012 eq. 4: quadratic in range, with a
/// hyperbolic term that blows up as the surface turns edge-on.
float AxialNoiseSigma(float depth, float incidenceRadians)
{
	// 1. Axial term. Quadratic in range about sigmaOffsetMeters.
	float fromOffset = depth - g_sigmaOffsetMeters;
	float sigma = g_sigmaConstant + g_sigmaQuadratic * fromOffset * fromOffset;

	// 2. Incidence term. The paper's denominator vanishes at exactly 90 degrees, so the angle is
	//    held just below it -- a grazing observation must come out very uncertain, not infinite.
	if (g_sigmaAngular > 0.0 && depth > 0.0)
	{
		const float kHalfPi = 1.5707963;
		float theta = clamp(incidenceRadians, 0.0, kHalfPi - 0.087266); // <= 85 degrees
		float toGrazing = kHalfPi - theta;
		sigma += (g_sigmaAngular / sqrt(depth)) * (theta * theta) / (toGrazing * toGrazing);
	}
	return sigma;
}

/// Band half-width for one observation. Clamped BELOW by the voxel grid, because a band narrower
/// than a voxel admits no sample and the surface disappears instead of sharpening, and ABOVE by
/// the build-time truncation, which stays the normalizer every reader denormalizes with.
float AdaptiveBandWidth(float depth, float incidenceRadians, float voxelSize, float truncateDistance)
{
	if (g_bandSigmaMultiplier <= 0.0) return truncateDistance;
	float sigma = AxialNoiseSigma(depth, incidenceRadians);
	return clamp(g_bandSigmaMultiplier * sigma, g_bandMinimumVoxels * voxelSize, truncateDistance);
}

/// *********************************************
/// Integrate
/// *********************************************
void Integrate(
	vec3 point,
	vec3 unitNormal,
	vec3 camera,
	vec3 rayDirection,
	float depth,
	float voxelSize,
	float truncateDistance,
	float viewReliabilityFactor)
{
	// 1. Select the dominant direction layers for this surface normal.
	uint descDirection[3];
	float reliability[3];
	int dirCount = selectDirections(
		unitNormal,
		descDirection,
		reliability);

	// 2. March the truncation band and integrate each voxel.
	// truncateDistance == acceptable band width
	//
	// The march must follow the SAME axis the band membership test below measures along, or the
	// band is scaled by the angle between the two. Point-to-plane measures along the normal:
	// dot(rayDirection, unitNormal) == -cos(incidence), so marching along the ray reaches only
	// steps*voxelSize*cos(incidence) of normal offset and the +-truncation band is clipped by the
	// incidence cosine -- to half its depth by 60 deg and a third by 75 deg (measured; see
	// AdvancedTSDF.PointToPlaneBandFillsTheFullTruncationDepthAtEveryIncidence). The surface
	// POSITION survives (the reachable band stays symmetric in t) but the far band is lost, so the
	// same surface fuses differently depending on the angle it was approached from -- which is
	// exactly what a hand-held sweep does. Projective measures along the ray, so it marches along
	// the ray. One axis per mode.
	//
	// This also makes integrate agree with extraction: sumN accumulates unitNormal, and n IS the
	// exact gradient of the point-to-plane value dot(x - p, n), which is the slope the extract
	// kernel's Hermite branch assumes.
	bool usePointToPlane = (g_pointToPlane != 0u);
	vec3 marchDirection  = usePointToPlane ? unitNormal : rayDirection;

	// The band width follows the sensor's own uncertainty at this range and incidence; the stored
	// value below is still normalized by truncateDistance, so every reader's `tsdf * truncation`
	// keeps meaning what it did.
	float incidence = acos(clamp(dot(unitNormal, -rayDirection), 0.0, 1.0));
	float bandWidth = AdaptiveBandWidth(depth, incidence, voxelSize, truncateDistance);
	int  steps = int(ceil(bandWidth / voxelSize)) + 1;
	for (int t = -steps; t <= steps; t++) {

		// 2.1. Signed distance from this voxel centre to the surface.
		//      point-to-plane (normal-based) removes the grazing-angle bias of the
		//      projective ray distance; both are positive on the camera-facing side.
		vec3  samplePos   = point + marchDirection * (float(t) * voxelSize);
		ivec3 voxel       = ivec3(floor(samplePos / voxelSize));
		vec3  voxelCenter = (vec3(voxel) + vec3(0.5)) * voxelSize;
		float voxel2point = usePointToPlane
			? dot(voxelCenter - point, unitNormal)
			: depth - dot(voxelCenter - camera, rayDirection);
		if (abs(voxel2point) > bandWidth) continue;
		float tsdf = clamp(voxel2point / truncateDistance, -1.0, 1.0);

		// Observation confidence for this band voxel. Two profiles:
		//
		//  * Behind-surface drop-off (Bylow et al. 2013; Voxblox eq. 5). The front of the band is
		//    free space the sensor looked THROUGH -- it was observed. The back is occluded and
		//    nothing has ever seen it, so its value is an extrapolation from the surface. Full
		//    weight in front and for one voxel behind, then a linear ramp to zero at the
		//    truncation. Voxblox uses exactly this with epsilon = v and delta = 4v.
		//  * A1 (the default): symmetric in |tsdf|, which discounts the observed side of the band
		//    as much as the occluded one.
		float confidence;
		if (g_behindSurfaceDropoff != 0u)
		{
			float epsilon = voxelSize;
			confidence = (voxel2point > -epsilon)
				? 1.0
				: max(0.0, (voxel2point + truncateDistance) / max(truncateDistance - epsilon, 1e-9));
		}
		else
		{
			confidence = 1.0 - g_confWeight * abs(tsdf);
		}

		// 2.2. Accumulate the weighted TSDF and the observed normal (stored gradient)
		//      into every selected direction layer.
		for (int di = 0; di < dirCount; di++) {
			float w = viewReliabilityFactor * reliability[di] * confidence;
			if (w <= 0.0) continue;

			uint key;
			if (!packDirKey(voxel, descDirection[di], key)) continue;

			uint slot = findOrInsert(key);
			if (slot == HASH_INSERT_FAILED) { atomicAdd(g_insertFailureCount, 1u); continue; }
			atomicAdd(g_hash[slot].sumDW, int(tsdf * w * TSDF_SCALE));
			atomicAdd(g_hash[slot].sumW,  uint(w * TSDF_SCALE));
			atomicAdd(g_hash[slot].sumNx, int(unitNormal.x * w * TSDF_SCALE));
			atomicAdd(g_hash[slot].sumNy, int(unitNormal.y * w * TSDF_SCALE));
			atomicAdd(g_hash[slot].sumNz, int(unitNormal.z * w * TSDF_SCALE));
		}
	}
}

void main()
{
	uint gid = gl_GlobalInvocationID.x;
	if (gid >= g_numPoints) return;

	vec3 point = vec3(
		g_points[gid * 3u],
		g_points[gid * 3u + 1u],
		g_points[gid * 3u + 2u]);

	// Windowing filter
	// - 포인트들이 해당 TSDF(Window)에 속하는지 검사한다.
	// point to window coordinates
	ivec3 localVoxel = ivec3(floor(point / g_voxelSize)) - ivec3(g_originX, g_originY, g_originZ);
	if (localVoxel.x < 0 || localVoxel.x > 511 ||
		localVoxel.y < 0 || localVoxel.y > 511 ||
		localVoxel.z < 0 || localVoxel.z > 511) return;

	vec3 normal = vec3(
		g_normals[gid * 3u],
		g_normals[gid * 3u + 1u],
		g_normals[gid * 3u + 2u]);
	vec3 camera = vec3(g_camX, g_camY, g_camZ);

	float normalLength = length(normal);
	if (normalLength < 1e-6) return;
	vec3 unitNormal = normal / normalLength;

	vec3  cam2point = point - camera;
	float depth     = length(cam2point);
	if (depth < 1e-6) return;
	vec3 rayDirection = cam2point / depth;

	// View-angle confidence: a back-facing/grazing observation (<= 0) contributes nothing.
	float viewReliabilityFactor = (g_viewAngleWeight != 0u)
		? max(0.0, dot(unitNormal, -rayDirection))
		: 1.0;
	if (viewReliabilityFactor <= 0.0) return;

	Integrate(
		point,
		unitNormal,
		camera,
		rayDirection,
		depth,
		g_voxelSize,
		g_truncation,
		viewReliabilityFactor);
}
