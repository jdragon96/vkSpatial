#version 460
layout(local_size_x = 64) in;

#define TSDF_SCALE  10000
#define MIN_WEIGHT  (TSDF_SCALE / 2)

#include "voxel_common.glsl"   // EMPTY_KEY, MAX_PROBE, wangHash

struct DirEntry {
    uint key;
    int  weightedDistanceSum;  // sum of (signed distance * weight)
    uint weightSum;            // sum of weights
    int  sumNx;                // stored gradient (Σ n·w·SCALE)
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
};

// Pack a world voxel coordinate + direction layer into a 32-bit hash key.
// Returns false when the voxel lies outside the 512^3 movable window.
bool packDirKey(ivec3 voxel, uint direction, out uint key) {
    ivec3 localVoxel = voxel - ivec3(g_originX, g_originY, g_originZ);
    if (localVoxel.x < 0 || localVoxel.x > 511 ||
        localVoxel.y < 0 || localVoxel.y > 511 ||
        localVoxel.z < 0 || localVoxel.z > 511) {
        key = 0u;
        return false;
    }
    key = ((uint(localVoxel.x) & 0x1FFu) << 21u)
        | ((uint(localVoxel.y) & 0x1FFu) << 12u)
        | ((uint(localVoxel.z) & 0x1FFu) <<  3u)
        |  (direction & 0x7u);
    return true;
}

// Reverse of packDirKey: recover the world voxel coordinate and direction layer.
void unpackDirKey(uint key, out ivec3 voxel, out uint direction) {
    ivec3 localVoxel;
    localVoxel.x = int((key >> 21u) & 0x1FFu);
    localVoxel.y = int((key >> 12u) & 0x1FFu);
    localVoxel.z = int((key >>  3u) & 0x1FFu);
    direction = key & 0x7u;
    voxel = localVoxel + ivec3(g_originX, g_originY, g_originZ);
}

// Hash probe for (voxel, direction). Ported from voxel_tsdf_mc.comp's getTSDF: from wangHash(key),
// MAX_PROBE linear probes; false on EMPTY_KEY hit, weight below MIN_WEIGHT, or voxel outside the
// movable window.
bool fetchDirectionalValue(ivec3 voxel, uint direction, out float value) {
    value = 0.0;
    uint key;
    if (!packDirKey(voxel, direction, key)) return false;
    uint slot = wangHash(key) % g_hashCapacity;
    for (uint probe = 0u; probe < MAX_PROBE; probe++) {
        DirEntry entry = g_hash[(slot + probe) % g_hashCapacity];
        if (entry.key == EMPTY_KEY) return false;
        if (entry.key == key) {
            if (entry.weightSum < uint(MIN_WEIGHT)) return false;
            value = float(entry.weightedDistanceSum) / float(entry.weightSum);
            return true;
        }
    }
    return false;
}

int estimateCrossingPosition(ivec3 voxel, uint direction, float centerValue, out vec3 outPosition) {
    vec3 positionSum = vec3(0.0);
    int  crossingCount = 0;
    for (int axis = 0; axis < 3; axis++) {
        ivec3 neighborVoxel = voxel;
        neighborVoxel[axis] += 1;

        float neighborValue;
        if (!fetchDirectionalValue(neighborVoxel, direction, neighborValue)) continue;
        if ((centerValue > 0.0) == (neighborValue > 0.0)) continue;  // same sign: no crossing
        if (centerValue == neighborValue) continue;                  // guard divide-by-zero

        float interpolant = centerValue / (centerValue - neighborValue);
        vec3  crossingPosition = (vec3(voxel) + vec3(0.5)) * g_voxelSize;
        crossingPosition[axis] += interpolant * g_voxelSize;

        positionSum += crossingPosition;
        crossingCount++;
    }
    outPosition = (crossingCount > 0) ? positionSum / float(crossingCount) : vec3(0.0);
    return crossingCount;
}

// Estimate the surface normal from the central-difference gradient of the TSDF in the
// same direction layer, falling back to a one-sided difference when a neighbour is missing.
// Returns false when the gradient is too small to yield a stable direction.
bool estimateNormal(ivec3 voxel, uint direction, float centerValue, out vec3 outNormal) {
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
    if (gradientLength < 1e-6) {
        outNormal = vec3(0.0);
        return false;
    }
    outNormal = gradient / gradientLength;
    return true;
}

// Append one oriented surface point to the candidate buffer. The counter is always
// advanced (so g_count reflects total attempts); returns false when the buffer is full.
bool writeCandidate(vec3 position, vec3 normal) {
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

void main() 
{
    uint entryIndex = gl_GlobalInvocationID.x;
    if (entryIndex >= g_hashCapacity) return;

    DirEntry entry = g_hash[entryIndex];
    if (entry.key == EMPTY_KEY) return;
    if (entry.weightSum < uint(MIN_WEIGHT)) return;

    ivec3 voxel;
    uint  direction;
    unpackDirKey(entry.key, voxel, direction);
    float centerValue = float(entry.weightedDistanceSum) / float(entry.weightSum);

    vec3 surfacePosition;
    if (estimateCrossingPosition(voxel, direction, centerValue, surfacePosition) == 0) return;

    // mode-3 hybrid: stored-gradient normal (denoised), central-difference fallback.
    vec3 sumN = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
    float sumNlen = length(sumN);
    vec3 surfaceNormal;
    if (sumNlen > 1e-6) {
        surfaceNormal = sumN / sumNlen;
    } else if (!estimateNormal(voxel, direction, centerValue, surfaceNormal)) {
        return; // degenerate: no stored gradient and no finite-difference gradient
    }

    writeCandidate(surfacePosition, surfaceNormal);
}
