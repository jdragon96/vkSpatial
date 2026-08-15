#version 460

/// *********************************************
/// Constants
/// *********************************************
layout(local_size_x = 64) in;

#define TSDF_SCALE  10000
#define MIN_WEIGHT  (TSDF_SCALE / 2)

#include "voxel_common.glsl"   // EMPTY_KEY, MAX_PROBE, wangHash

// 24 bytes. Layout must match DirEntry in kernel_AdvancedTSDF.integrate.comp.glsl and AdvancedTSDF.h.
struct DirEntry
{
	uint key;
	int  sumDW;   // Σ tsdf · w · TSDF_SCALE
	uint sumW;    // Σ w · TSDF_SCALE
	int  sumNx;   // Σ n · w · TSDF_SCALE  (stored gradient)
	int  sumNy;
	int  sumNz;
};

layout(std430, set = 0, binding = 0) readonly buffer HashTable  { DirEntry g_hash[]; };
layout(std430, set = 0, binding = 1)          buffer Candidates { float g_cand[]; };
layout(        set = 0, binding = 2)          buffer Counter    { uint g_count; };

layout(push_constant) uniform PC {
	float g_voxelSize;
	uint  g_hashCapacity;
	uint  g_maxCandidates;
	int   g_originX;
	int   g_originY;
	int   g_originZ;
	float g_truncation;   // A2: needed to scale the stored-gradient derivative
	uint  g_hermite;      // A2: 1 = cubic-Hermite zero-crossing, 0 = linear
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

void unpackDirKey(uint key, out ivec3 voxel, out uint direction)
{
	ivec3 localVoxel;
	localVoxel.x = int((key >> 21u) & 0x1FFu);
	localVoxel.y = int((key >> 12u) & 0x1FFu);
	localVoxel.z = int((key >>  3u) & 0x1FFu);
	direction = key & 0x7u;
	voxel = localVoxel + ivec3(g_originX, g_originY, g_originZ);
}

#include "TSDF/Memory/Hash/HashStrategy.glsl"

// Hash lookup for (voxel, direction); false on HASH_NOT_FOUND, weight below MIN_WEIGHT, or
// voxel outside the movable window.
bool fetchDirectionalValue(ivec3 voxel, uint direction, out float value)
{
	value = 0.0;
	uint key;
	if (!packDirKey(voxel, direction, key)) return false;
	uint slot = findSlot(key);
	if (slot == HASH_NOT_FOUND) return false;
	DirEntry entry = g_hash[slot];
	if (entry.sumW < uint(MIN_WEIGHT)) return false;
	value = float(entry.sumDW) / float(entry.sumW);
	return true;
}

// A2: hash lookup returning both value and stored-gradient normal (for Hermite interpolation).
bool fetchDirectionalValueAndNormal(ivec3 voxel, uint direction, out float value, out vec3 normal)
{
	value = 0.0;
	normal = vec3(0.0);
	uint key;
	if (!packDirKey(voxel, direction, key)) return false;
	uint slot = findSlot(key);
	if (slot == HASH_NOT_FOUND) return false;
	DirEntry entry = g_hash[slot];
	if (entry.sumW < uint(MIN_WEIGHT)) return false;
	value = float(entry.sumDW) / float(entry.sumW);
	vec3 sumN = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
	float len = length(sumN);
	normal = (len > 1e-6) ? sumN / len : vec3(0.0);
	return true;
}

// A2: root of the cubic Hermite through endpoints (0: value c0, slope g0) and (1: c1, g1),
// Newton from the linear seed. g = d(value)/d(param) = n_axis · voxel / truncation.
float hermiteRoot(float c0, float c1, float g0, float g1)
{
	float t = c0 / (c0 - c1); // linear seed
	for (int it = 0; it < 3; it++) {
		float t2 = t * t, t3 = t2 * t;
		float p  = (2.0*t3 - 3.0*t2 + 1.0) * c0 + (t3 - 2.0*t2 + t) * g0
		         + (-2.0*t3 + 3.0*t2) * c1 + (t3 - t2) * g1;
		float dp = (6.0*t2 - 6.0*t) * c0 + (3.0*t2 - 4.0*t + 1.0) * g0
		         + (-6.0*t2 + 6.0*t) * c1 + (3.0*t2 - 2.0*t) * g1;
		if (abs(dp) < 1e-8) break;
		t = clamp(t - p / dp, 0.0, 1.0);
	}
	return t;
}

/// *********************************************
/// Surface position
/// *********************************************
// Estimate the surface position by scanning the three +axis neighbours in the same
// direction layer for zero crossings and averaging their interpolated positions.
// Returns the number of crossings; outPosition holds the average when non-zero.
int estimateCrossingPosition(ivec3 voxel, uint direction, float centerValue, vec3 centerNormal,
                             out vec3 outPosition)
{
	vec3 positionSum = vec3(0.0);
	int  crossingCount = 0;
	for (int axis = 0; axis < 3; axis++) {
		ivec3 neighborVoxel = voxel;
		neighborVoxel[axis] += 1;

		float neighborValue;
		vec3  neighborNormal;
		if (!fetchDirectionalValueAndNormal(neighborVoxel, direction, neighborValue, neighborNormal)) continue;
		if ((centerValue > 0.0) == (neighborValue > 0.0)) continue;  // same sign: no crossing
		if (centerValue == neighborValue) continue;                  // guard divide-by-zero

		float interpolant;
		if (g_hermite != 0u) {
			// A2 gradient-augmented (cubic Hermite): slope = n_axis · voxel / truncation.
			float g0 = centerNormal[axis]   * g_voxelSize / g_truncation;
			float g1 = neighborNormal[axis] * g_voxelSize / g_truncation;
			interpolant = hermiteRoot(centerValue, neighborValue, g0, g1);
		} else {
			interpolant = centerValue / (centerValue - neighborValue);
		}
		vec3 crossingPosition = (vec3(voxel) + vec3(0.5)) * g_voxelSize;
		crossingPosition[axis] += interpolant * g_voxelSize;

		positionSum += crossingPosition;
		crossingCount++;
	}
	outPosition = (crossingCount > 0) ? positionSum / float(crossingCount) : vec3(0.0);
	return crossingCount;
}

/// *********************************************
/// Surface normal
/// *********************************************
// Fallback only: central-difference gradient of the TSDF in the same direction layer
// (one-sided when a neighbour is missing). Used when the stored gradient is degenerate.
bool estimateNormalFallback(ivec3 voxel, uint direction, float centerValue, out vec3 outNormal)
{
	vec3 gradient = vec3(0.0);
	for (int axis = 0; axis < 3; axis++) {
		ivec3 forwardVoxel  = voxel; forwardVoxel[axis]  += 1;
		ivec3 backwardVoxel = voxel; backwardVoxel[axis] -= 1;

		float forwardValue, backwardValue;
		bool hasForward  = fetchDirectionalValue(forwardVoxel,  direction, forwardValue);
		bool hasBackward = fetchDirectionalValue(backwardVoxel, direction, backwardValue);

		if (hasForward && hasBackward) gradient[axis] = (forwardValue - backwardValue) * 0.5;
		else if (hasForward)           gradient[axis] = forwardValue - centerValue;
		else if (hasBackward)          gradient[axis] = centerValue - backwardValue;
	}
	float gradientLength = length(gradient);
	if (gradientLength < 1e-6)
	{
		outNormal = vec3(0.0);
		return false;
	}
	outNormal = gradient / gradientLength;
	return true;
}

/// *********************************************
/// Output
/// *********************************************
// Append one oriented surface point to the candidate buffer. The counter is always
// advanced (so g_count reflects total attempts); returns false when the buffer is full.
bool writeCandidate(vec3 position, vec3 normal)
{
	uint outputIndex = atomicAdd(g_count, 1u);
	if (outputIndex >= g_maxCandidates) return false;

	uint base = outputIndex * 6u;
	g_cand[base + 0u] = position.x;
	g_cand[base + 1u] = position.y;
	g_cand[base + 2u] = position.z;
	g_cand[base + 3u] = normal.x;
	g_cand[base + 4u] = normal.y;
	g_cand[base + 5u] = normal.z;
	return true;
}

/// *********************************************
/// Extract (mode-3 hybrid)
/// *********************************************
void main()
{
	// tiles are sorted previously
	uint entryIndex = gl_GlobalInvocationID.x;
	if (entryIndex >= g_hashCapacity) return;

	DirEntry entry = g_hash[entryIndex];
	if (entry.key == EMPTY_KEY) return;
	if (entry.sumW < uint(MIN_WEIGHT)) return;

	ivec3 voxel;
	uint  direction;
	unpackDirKey(entry.key, voxel, direction);
	float centerValue = float(entry.sumDW) / float(entry.sumW);

	// Stored-gradient normal (denoised, mode-3); also feeds the A2 Hermite position slope.
	vec3  storedGradient = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
	float gradientLength = length(storedGradient);
	vec3  centerNormal = (gradientLength > 1e-6) ? storedGradient / gradientLength : vec3(0.0);

	// 1. Position: zero-crossing interpolation (linear, or A2 gradient-augmented Hermite).
	vec3 surfacePosition;
	if (estimateCrossingPosition(voxel, direction, centerValue, centerNormal, surfacePosition) == 0)
		return;

	// 2. Normal: stored gradient, central-difference fallback when degenerate.
	vec3 surfaceNormal;
	if (gradientLength > 1e-6) {
		surfaceNormal = centerNormal;
	} else if (!estimateNormalFallback(voxel, direction, centerValue, surfaceNormal)) {
		return; // degenerate: no stored gradient and no finite-difference gradient
	}

	writeCandidate(surfacePosition, surfaceNormal);
}
