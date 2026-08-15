#version 460
layout(local_size_x = 256) in;

// Dominant-direction TSDF integration (design doc §13). Extends SimpleTSDF's
// voxel_tsdf_integrate.comp ray-march: each sample writes only into the direction
// layer chosen by its normal's dominant axis, and only into resident groups
// (invariant #7 — indexGrid miss → skip). Touched slots are marked dirty.

#define LOCAL_GRID 50
#define TSDF_SCALE 10000.0

struct PointSample { float px, py, pz, nx, ny, nz; };
struct GpuVoxel { int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; };
struct ActiveGroupMeta { int gx; int gy; int gz; uint packed; };

layout(push_constant) uniform PC {
    uint  g_numPoints;
    float g_voxelSize;
    float g_truncation;
    int   g_baseX;
    int   g_baseY;
    int   g_baseZ;
    float g_camX;
    float g_camY;
    float g_camZ;
    uint  g_maxDirections;
    uint  g_dirExponent;
    uint  g_viewAngleWeight;
    uint  g_pointToPlane;   // A1: 0 = projective SDF, 1 = point-to-plane SDF
};

layout(std430, set = 0, binding = 0) readonly buffer Points { PointSample g_points[]; };
layout(std430, set = 0, binding = 1) readonly buffer IndexGrid { uint g_indexGrid[]; };
layout(std430, set = 0, binding = 2) buffer Pool { GpuVoxel g_pool[]; };
layout(std430, set = 0, binding = 3) buffer Meta { ActiveGroupMeta g_meta[]; };

// Must match dominantAxisOf() in DirectionalTSDF.cpp exactly (comparison order + ties).
// Kept for reference; superseded by topK() below (which reduces to this for K=1).
uint dominantAxis(vec3 n) {
    vec3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) return n.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.x && a.y >= a.z) return n.y >= 0.0 ? 2u : 3u;
    return n.z >= 0.0 ? 4u : 5u;
}

// integer power via repeated multiply (matches CPU ipow exactly)
float ipow(float x, uint p){ float r=1.0; for(uint i=0u;i<p;i++) r*=x; return r; }

// returns count; fills dirs[]/rel[]. Mirrors TopKDirections() in DirectionalTSDF.cpp.
int topK(vec3 n, out uint dirs[3], out float rel[3]) {
    float a[3] = float[](abs(n.x), abs(n.y), abs(n.z));
    uint  d[3] = uint[]( n.x>=0.0?0u:1u, n.y>=0.0?2u:3u, n.z>=0.0?4u:5u );
    // STABLE insertion sort desc by a[], preserving x,y,z seed order on ties
    // (must match TopKDirections in DirectionalTSDF.cpp EXACTLY — shift only on strict <).
    for(int i=1;i<3;i++){ float ka=a[i]; uint kd=d[i]; int j=i-1;
        while(j>=0 && a[j]<ka){ a[j+1]=a[j]; d[j+1]=d[j]; j--; }
        a[j+1]=ka; d[j+1]=kd; }
    float rmax = ipow(a[0], g_dirExponent);
    int cnt=0; uint K = g_maxDirections<1u?1u:g_maxDirections;
    for(int i=0;i<3 && uint(cnt)<K;i++){
        float rd = ipow(a[i], g_dirExponent);
        float r = rmax>0.0 ? rd/rmax : 0.0;
        if(i==0){ dirs[cnt]=d[0]; rel[cnt]=1.0; cnt++; continue; }
        if(r>=0.05){ dirs[cnt]=d[i]; rel[cnt]=r; cnt++; }
    }
    return cnt;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_numPoints) return;

    PointSample s = g_points[gid];
    vec3 p = vec3(s.px, s.py, s.pz);
    vec3 cam = vec3(g_camX, g_camY, g_camZ);

    vec3 diff = p - cam;
    float depth = length(diff);
    if (depth < 1e-6) return;
    vec3 rayDir = diff / depth;

    vec3 nrm = vec3(s.nx, s.ny, s.nz);
    uint dirs[3]; float rel[3];
    int nd = topK(nrm, dirs, rel);
    float viewFactor = (g_viewAngleWeight != 0u) ? max(0.0, dot(nrm, -rayDir)) : 1.0;

    int steps = int(ceil(g_truncation / g_voxelSize)) + 1;
    for (int t = -steps; t <= steps; t++) {
        vec3 samplePos = p + rayDir * (float(t) * g_voxelSize);
        ivec3 v = ivec3(floor(samplePos / g_voxelSize));
        vec3 vCenter = (vec3(v) + vec3(0.5)) * g_voxelSize;

        // A1 experiment: projective ray distance vs point-to-plane (normal-based).
        // point-to-plane removes the grazing-angle bias of the projective form.
        float sdf = (g_pointToPlane != 0u)
                        ? dot(vCenter - p, normalize(nrm))
                        : depth - dot(vCenter - cam, rayDir);
        if (abs(sdf) > g_truncation) continue;

        ivec3 g = v >> 3; // floor division by 8, negatives included
        int lx = g.x - g_baseX;
        int ly = g.y - g_baseY;
        int lz = g.z - g_baseZ;
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= LOCAL_GRID || ly >= LOCAL_GRID || lz >= LOCAL_GRID)
            continue;

        float newValue = clamp(sdf / g_truncation, -1.0, 1.0);

        for (int di = 0; di < nd; di++) {
            uint dir = dirs[di];
            uint cell = (((uint(lz)*uint(LOCAL_GRID)+uint(ly))*uint(LOCAL_GRID)+uint(lx))*6u) + dir;
            uint poolIndex = g_indexGrid[cell];
            if (poolIndex == 0xFFFFFFFFu) continue; // not resident → not in the write set
            ivec3 lv = v & 7;
            uint addr = poolIndex*512u + (uint(lv.z)*8u+uint(lv.y))*8u+uint(lv.x);
            float w = viewFactor * rel[di];
            atomicAdd(g_pool[addr].sumDW, int(newValue * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumW,  uint(w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumNx, int(nrm.x * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumNy, int(nrm.y * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumNz, int(nrm.z * w * TSDF_SCALE));
            atomicOr(g_meta[poolIndex].packed, 1u << 16);
        }
    }
}
