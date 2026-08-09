#version 450
/// One point-to-plane ICP iteration. Thread = source point. Transforms by push-constant T (centred
/// frame), grid-NN correspondence, then reduces the 6x6 normal equations H,b per-workgroup into shared
/// fixed-point ints (no MoltenVK float atomics); thread 0 writes this workgroup's 29-int partial.
///
/// Centring: both `src` and `tgt` are pre-shifted by -c (the target centroid) on the CPU before upload,
/// and `T` is applied in that SAME centred frame: p = T * (src_i - c). A common translation applied to
/// both p and q leaves the point-to-plane residual (p-q)*n invariant, but the Jacobian's rotational
/// block J_rot = p x n is NOT translation-invariant on its own -- centring must be applied consistently
/// to p (and hence to the cross product) everywhere, which this shader and the CPU reference both do.
/// So H,b computed here are the CENTRED-frame normal equations; they equal the CPU reference on the
/// SAME points/T (see GpuPointToPlaneIcp::Accumulate doc comment).
layout(local_size_x = 256) in;
const float SCALE = 10000.0;
// Fixed-point overflow bound: each workgroup sums up to 256 inliers' contributions into an int32
// `shared` accumulator (atomicAdd, see below). The largest term is the H rotation block
// J[r]*J[col]*SCALE where J[0..2] = p x n (|n| <= 1, so |p x n| <= |p|); worst case ~256 * |p|^2 * 1e4
// stays under INT32_MAX (~2.1e9) only while centroid-relative point magnitudes |p| stay within
// ~29 m (256 * 29^2 * 1e4 =~ 2.16e9). Fine for a cropped depth frame (centred on the target
// centroid, extent a few metres) but keep this in mind if maxCorrDist/crop sizes grow much larger.

// SCALAR push-constant fields ONLY (no vec3/ivec3): GLSL aligns vec3 to 16 bytes, which would NOT
// match the tightly-packed C++ IcpPC struct. Reconstruct vectors in main().
layout(push_constant) uniform PC {
    mat4  g_T;          // current pose (centred frame), column-major
    float g_originX, g_originY, g_originZ, g_cell;  // grid AABB min + cell size
    int   g_dimsX, g_dimsY, g_dimsZ;                // cells per axis
    float g_maxCorr;
    float g_huberScale;                 // robust-weight knee (world units)
    float g_normalCompatibilityCosine;  // reject correspondence if sourceN.targetN < this
    uint  g_numSrc, g_numCells;
};
layout(std430, binding=0) readonly buffer Src        { vec4 g_src[]; };        // xyz used (centred)
layout(std430, binding=1) readonly buffer TgtPts     { vec4 g_tgtPts[]; };     // centred
layout(std430, binding=2) readonly buffer TgtNrm     { vec4 g_tgtNrm[]; };
layout(std430, binding=3) readonly buffer BucketStart{ uint g_bstart[]; };
layout(std430, binding=4) readonly buffer BucketIdx  { uint g_bidx[]; };
layout(std430, binding=5) buffer Partials            { int g_part[]; };        // [numWG * 29]
layout(std430, binding=6) readonly buffer SrcNrm     { vec4 g_srcNrm[]; };     // xyz used (NOT centred)

shared int s_acc[29];

int cellIndex(ivec3 c, ivec3 dims) { return (c.z * dims.y + c.y) * dims.x + c.x; }

void main() {
    vec3  g_origin = vec3(g_originX, g_originY, g_originZ);
    ivec3 g_dims   = ivec3(g_dimsX, g_dimsY, g_dimsZ);

    uint tid = gl_LocalInvocationID.x;
    if (tid < 29u) s_acc[tid] = 0;
    barrier();

    uint i = gl_GlobalInvocationID.x;
    if (i < g_numSrc) {
        vec3 p = (g_T * vec4(g_src[i].xyz, 1.0)).xyz;
        ivec3 c = ivec3(floor((p - g_origin) / g_cell));
        int best = -1; float bestD2 = g_maxCorr * g_maxCorr;
        for (int dz=-1; dz<=1; ++dz) for (int dy=-1; dy<=1; ++dy) for (int dx=-1; dx<=1; ++dx) {
            ivec3 cc = c + ivec3(dx,dy,dz);
            if (any(lessThan(cc, ivec3(0))) || any(greaterThanEqual(cc, g_dims))) continue;
            int ci = cellIndex(cc, g_dims);
            for (uint k = g_bstart[ci]; k < g_bstart[ci+1]; ++k) {
                uint idx = g_bidx[k];
                float d2 = dot(p - g_tgtPts[idx].xyz, p - g_tgtPts[idx].xyz);
                if (d2 < bestD2) { bestD2 = d2; best = int(idx); }
            }
        }
        if (best >= 0) {
            vec3 q = g_tgtPts[best].xyz;
            vec3 n = g_tgtNrm[best].xyz;
            // Normal-compatibility rejection: drop the correspondence when the (pose-rotated) source
            // normal disagrees too much with the target normal, even though it passed the distance
            // gate above -- same rejection the CPU reference (AlignPointToPlaneIcp) applies.
            vec3 transformedSourceNormal = mat3(g_T) * g_srcNrm[i].xyz;
            if (dot(transformedSourceNormal, n) < g_normalCompatibilityCosine) {
                // rejected: contributes nothing (fall through, no atomics)
            } else {
                float e = dot(p - q, n);
                float J[6];
                vec3 pxn = cross(p, n);
                J[0]=pxn.x; J[1]=pxn.y; J[2]=pxn.z; J[3]=n.x; J[4]=n.y; J[5]=n.z;
                // Huber (robust) weight: full trust inside the knee, down-weighted beyond it, so a few
                // gross outliers cannot dominate the normal equations -- same weighting the CPU
                // reference applies. sum of squared residuals (slot 28) stays UNWEIGHTED (a true fit
                // metric), matching AlignPointToPlaneIcp.
                float absoluteResidual = abs(e);
                float robustWeight = absoluteResidual <= g_huberScale ? 1.0 : g_huberScale / absoluteResidual;
                int k = 0;                                   // upper-triangular H (row-major, 21 entries)
                for (int r=0; r<6; ++r) for (int col=r; col<6; ++col)
                    atomicAdd(s_acc[k++], int(round(robustWeight*J[r]*J[col]*SCALE)));
                for (int r=0; r<6; ++r)
                    atomicAdd(s_acc[21+r], int(round(robustWeight*(-J[r]*e)*SCALE)));
                atomicAdd(s_acc[27], 1);                      // inlier count: raw +1, NOT scaled
                atomicAdd(s_acc[28], int(round(e * e * SCALE))); // sum of squared residuals (fixed-point, unweighted)
            }
        }
    }
    barrier();
    if (tid < 29u) g_part[gl_WorkGroupID.x * 29u + tid] = s_acc[tid];
}
