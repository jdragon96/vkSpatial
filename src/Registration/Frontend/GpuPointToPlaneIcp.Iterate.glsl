#version 450
layout(local_size_x = 256) in;
const float SCALE = 10000.0;
layout(push_constant) uniform PC {
    mat4  g_T;          // current pose (centred frame), column-major
    float g_originX, g_originY, g_originZ, g_cell;  // grid AABB min + cell size
    int   g_dimsX, g_dimsY, g_dimsZ;                // cells per axis
    float g_maxCorr;    // per-ITERATION distance filter -- may be NARROWER than g_cell (coarse-to-fine
                        // annealing: the grid is built once, at the WIDEST distance, and only this
                        // filter shrinks per iteration; the 3x3x3 neighbor scan below still covers any
                        // radius <= g_cell, so correctness holds for every annealed value)
    float g_huberScale;                 // robust-weight knee (world units)
    float g_normalCompatibilityCosine;  // reject correspondence if sourceN.targetN < this
    uint  g_numSrc, g_numCells;
};
layout(std430, binding=0) readonly buffer Src        { vec4 g_src[]; };        // xyz used (centred)
layout(std430, binding=1) readonly buffer TgtPts     { vec4 g_tgtPts[]; };     // centred
layout(std430, binding=2) readonly buffer TgtNrm     { vec4 g_tgtNrm[]; };
layout(std430, binding=3) readonly buffer BucketStart{ uint g_bstart[]; };
layout(std430, binding=4) readonly buffer BucketIdx  { uint g_bidx[]; };
layout(std430, binding=5) buffer Partials            { int g_part[]; };        // [numWG * 29]: slots 0..27 fixed-point int, slot 28 float bits (residual sum)
layout(std430, binding=6) readonly buffer SrcNrm     { vec4 g_srcNrm[]; };     // xyz used (NOT centred)

shared int   s_acc[28];       // H (21) + b (6) + inlier count, fixed-point atomics
// Squared residuals reduce in FLOAT, per thread then tree -- the fixed-point path (int(round(e*e*SCALE)))
// zeroed every term with |e| under ~7 mm, so any rmse below that floor read as noise. A shared-memory
// tree needs no float atomics and its addition order is fixed, so replay determinism is preserved.
shared float s_squaredResiduals[256];

int cellIndex(ivec3 c, ivec3 dims) { return (c.z * dims.y + c.y) * dims.x + c.x; }

void main() {
    vec3  g_origin = vec3(g_originX, g_originY, g_originZ);
    ivec3 g_dims   = ivec3(g_dimsX, g_dimsY, g_dimsZ);

    uint tid = gl_LocalInvocationID.x;
    if (tid < 28u) s_acc[tid] = 0;
    barrier();

    float squaredResidual = 0.0;
    uint i = gl_GlobalInvocationID.x;
    if (i < g_numSrc) {
        // Centroid-Local to Centroid-World
        vec3 p = (g_T * vec4(g_src[i].xyz, 1.0)).xyz;
        // Centroid-World to Voxel
        ivec3 c = ivec3(floor((p - g_origin) / g_cell));
        int best = -1;
        float bestD2 = g_maxCorr * g_maxCorr;

        /// 1. Pre-pass over the query's OWN cell, for a tight rejection bound only -- `best` is not
        ///    touched here. The 3x3x3 block is a cube of side 3*maxCorr, 6.4x the volume of the
        ///    maxCorr sphere actually being searched, so nearly every cell is dead weight once any
        ///    close point is known. The home cell is the 14th of 27 in the scan order below, which
        ///    left the first thirteen to run against the loose starting bound.
        float pruneBound = bestD2;
        if (all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, g_dims)))
        {
            int homeCell = cellIndex(c, g_dims);
            for (uint k = g_bstart[homeCell]; k < g_bstart[homeCell + 1]; ++k)
            {
                float d2 = dot(p - g_tgtPts[k].xyz, p - g_tgtPts[k].xyz);
                pruneBound = min(pruneBound, d2);
            }
        }

        /// 2. Scan every cell in the original order, so the winner is decided exactly as before.
        ///    Both rejections are safe against ties: a cell whose closest possible point is at
        ///    d2 >= bestD2 holds nothing the strict `<` below would accept, and one at
        ///    d2 > pruneBound holds nothing that can beat the home cell's own best, which the
        ///    final bestD2 is already at or below. STRICT here -- rejecting on `>=` would drop a
        ///    point that equals pruneBound but comes first, and first-wins is the original rule.
        for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            ivec3 cc = c + ivec3(dx,dy,dz);
            if (any(lessThan(cc, ivec3(0))) || any(greaterThanEqual(cc, g_dims))) continue;

            vec3 cellMin = g_origin + vec3(cc) * g_cell;
            vec3 cellDelta = max(max(cellMin - p, vec3(0.0)), p - (cellMin + vec3(g_cell)));
            float cellD2 = dot(cellDelta, cellDelta);
            if (cellD2 >= bestD2 || cellD2 > pruneBound) continue;

            // The target is uploaded permuted into bucket order (see prepareCentred), so a cell's
            // points are contiguous and the slot IS the target index -- no indirection.
            int ci = cellIndex(cc, g_dims);
            for (uint k = g_bstart[ci]; k < g_bstart[ci+1]; ++k) {
                float d2 = dot(p - g_tgtPts[k].xyz, p - g_tgtPts[k].xyz);
                if (d2 < bestD2) { bestD2 = d2; best = int(k); }
            }
        }
        if (best >= 0) {
            vec3 q = g_tgtPts[best].xyz;
            vec3 n = g_tgtNrm[best].xyz;
            vec3 transformedSourceNormal = mat3(g_T) * g_srcNrm[i].xyz;
            if (dot(transformedSourceNormal, n) < g_normalCompatibilityCosine) {
            } else {
                float e = dot(p - q, n);
                float J[6];
                vec3 pxn = cross(p, n);
                J[0]=pxn.x; J[1]=pxn.y; J[2]=pxn.z; J[3]=n.x; J[4]=n.y; J[5]=n.z;
                float absoluteResidual = abs(e);
                float robustWeight = absoluteResidual <= g_huberScale ? 1.0 : g_huberScale / absoluteResidual;
                int k = 0;
                for (int r=0; r<6; ++r) for (int col=r; col<6; ++col)
                    atomicAdd(s_acc[k++], int(round(robustWeight*J[r]*J[col]*SCALE)));
                for (int r=0; r<6; ++r)
                    atomicAdd(s_acc[21+r], int(round(robustWeight*(-J[r]*e)*SCALE)));
                atomicAdd(s_acc[27], 1);
                squaredResidual = e * e;
            }
        }
    }
    s_squaredResiduals[tid] = squaredResidual;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) s_squaredResiduals[tid] += s_squaredResiduals[tid + stride];
        barrier();
    }
    if (tid < 28u) g_part[gl_WorkGroupID.x * 29u + tid] = s_acc[tid];
    if (tid == 28u) g_part[gl_WorkGroupID.x * 29u + 28u] = floatBitsToInt(s_squaredResiduals[0]);
}
