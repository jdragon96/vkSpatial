#version 460

/// *********************************************
/// Constants
/// *********************************************
layout(local_size_x = 256) in;

#define TSDF_SCALE           10000.0
#define MIN_RELATIVE_WEIGHT  0.05   // drop directions weaker than 5% of the dominant one

#include "voxel_common.glsl"   // EMPTY_KEY, MAX_PROBE, wangHash

// 24 bytes: distance/weight accumulators + stored-gradient (observed normal) accumulators.
// Layout must match DirEntry in advanced_tsdf_extract.vert.glsl and AdvancedTSDFTypes.h.
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

uint findOrInsert(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint p = 0u; p < MAX_PROBE; p++) {
		uint index = (slot + p) % g_hashCapacity;
		uint prev = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
		if (prev == EMPTY_KEY)
		{
			atomicAdd(g_filledCount, 1u);
			return index;
		}
		if (prev == key) return index;
	}
	return ~0u;
}

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
		float a = align[i]; uint ax = axis[i];
		int j = i - 1;
		while (j >= 0 && align[j] < a)
		{
			align[j + 1] = align[j];
			axis[j + 1] = axis[j];
			j--;
		}
		align[j + 1] = a; axis[j + 1] = ax;
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
/// Integrate
/// *********************************************
void Integrate(
	vec3 point,
	vec3 unitNormal,
	vec3 camera,
	vec3 rayDirection,
	float depth,
	float voxelSize,
	float truncateFactor,
	float viewReliabilityFactor)
{
	// 1. Select the dominant direction layers for this surface normal.
	uint descDirection[3];
	float reliability[3];
	int dirCount = selectDirections(
		unitNormal,
		descDirection,
		reliability);

	// 2. March the truncation band along the ray and integrate each voxel.
	bool usePointToPlane = (g_pointToPlane != 0u);
	int  steps = int(ceil(truncateFactor / voxelSize)) + 1;
	for (int t = -steps; t <= steps; t++) {

		// 2.1. Signed distance from this voxel centre to the surface.
		//      point-to-plane (normal-based) removes the grazing-angle bias of the
		//      projective ray distance; both are positive on the camera-facing side.
		vec3  samplePos   = point + rayDirection * (float(t) * voxelSize);
		ivec3 voxel       = ivec3(floor(samplePos / voxelSize));
		vec3  voxelCenter = (vec3(voxel) + vec3(0.5)) * voxelSize;
		float voxel2point = usePointToPlane
			? dot(voxelCenter - point, unitNormal)
			: depth - dot(voxelCenter - camera, rayDirection);
		if (abs(voxel2point) > truncateFactor) continue;
		float tsdf = clamp(voxel2point / truncateFactor, -1.0, 1.0);

		// 2.2. Accumulate the weighted TSDF and the observed normal (stored gradient)
		//      into every selected direction layer.
		for (int di = 0; di < dirCount; di++) {
			float w = viewReliabilityFactor * reliability[di];
			if (w <= 0.0) continue;

			uint key;
			if (!packDirKey(voxel, descDirection[di], key)) continue;

			uint slot = findOrInsert(key);
			if (slot == ~0u) continue;
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
	vec3 normal = vec3(
		g_normals[gid * 3u],
		g_normals[gid * 3u + 1u],
		g_normals[gid * 3u + 2u]);
	vec3 camera = vec3(g_camX, g_camY, g_camZ);

	// Degenerate normal → the point carries no orientation; skip it (also guards
	// normalize() and the point-to-plane distance against a zero-length normal).
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
