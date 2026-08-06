# GPU Point-to-Plane ICP Tracker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU point-to-plane ICP tracker with a GPU-accelerated one whose per-frame cost is dominated by GPU work, so `--tracker icp` stays real-time as the map grows.

**Architecture:** A new self-contained `GpuPointToPlaneIcp` (owns an `Engine::Core::Context` + one compute pipeline) mirrors the CPU `AlignPointToPlaneIcp`. Per solve: the CPU crops the target to the source AABB, builds a compact uniform grid, and centres both clouds on the target centroid (numerical conditioning); a GPU kernel (`icp_iterate.comp.glsl`) transforms each source point by the current pose, finds its nearest target within `maxCorrDist` via the grid, and reduces the point-to-plane 6×6 normal equations `H`,`b` per-workgroup; the CPU sums the per-workgroup partials, solves the 6×6 with Eigen LDLT, updates the pose, and iterates. The `GpuIcpTracker` wraps it and registers as `"icp"` (the CPU one stays as `"icp-cpu"` for A/B).

**Tech Stack:** C++17, Vulkan compute (MoltenVK), GLSL compute shaders (runtime-compiled), Eigen.

## Global Constraints

- **No GPU float atomics** (MoltenVK lacks `VK_EXT_shader_atomic_float`). Reduce `H`,`b` **per-workgroup into `shared int` fixed-point accumulators** (magnitudes stay small within one 256-point workgroup after centring), write each workgroup's partial to its own global slot (no cross-workgroup atomics), and sum the partials on the CPU as `double`. Fixed-point scale = `10000.0` (matches `TSDF_SCALE` in the advanced_tsdf shaders).
- **Numerical conditioning:** subtract the target centroid `c` from both clouds before the GPU solve; solve in the centred frame; recover the world pose as `T_world = Translate(c) * T_centred * Translate(-c)`. This keeps `p×n` and `H` entries O(local-extent²), not O(world-position²), so fixed-point never overflows int32.
- **Shaders compile at runtime** from `VKBVH_SHADER_DIR` (`src/shader/`). A new `.comp.glsl` there needs **no CMake change** (see `AdvancedTSDF::Build` calling `m_kernel->Build("advanced_tsdf_integrate.comp.glsl")`).
- **CMake globs sources** (`Pipeline/*.cpp`, test `*.cpp`) — new `.cpp`/test files are auto-picked up, but require a **re-configure** (`cmake .` in the build dir) since GLOB is not `CONFIGURE_DEPENDS`.
- **Two build dirs:** `build/` = Debug (`-O0`, functional check), `build-rel/` = RelWithDebInfo (`-O3`, the **only** valid perf measurement — Eigen/CPU is 10–100× slower at `-O0`).
- **Reuse existing types** from `Engine/Registration/RegistrationTypes.h`: `PointCloud{points,normals}`, `RegistrationResult{T(Matrix4f),fitness,numInliers,valid}`, `RegistrationParam{maxIters,maxCorrDist,minInliers,convEps}`.
- **Everything is uncommitted** on branch `feature/dlp-structured-light`; commit each task locally (do not push to origin).
- **ComputePipeline API** (see `AdvancedTSDF.cpp`): `pipe.Build("x.comp.glsl").Bind(slot, buffer)…;` then per dispatch `pipe.Args(pcStruct); pipe.DispatchElements(numThreads);` (self-submits + `vkQueueWaitIdle`, synchronous). `Bind` is re-callable to rebind grown buffers.
- **Buffer API** (see `AdvancedTSDF.cpp` / `Buffer.h`): `Allocate(bytes)` (device-local), `AllocateHostVisible(bytes)`, `AllocateHostVisibleReadback(bytes)`, `MappedPtr()`, `FlushMapped(bytes)`, `InvalidateMapped(bytes)`, `Upload(ptr,bytes)`, `Download(ptr,bytes)`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/Engine/Pipeline/Registration/GpuIcp.h` (Create) | `GpuPointToPlaneIcp` class declaration + the CPU grid helper (`LocalGrid`). Lives in the Pipeline layer (which already links Core/Compute) so `Engine::Registration` (CPU/Ceres) stays GPU-free. |
| `src/Engine/Pipeline/Registration/GpuIcp.cpp` (Create) | Grid build, centring, upload, per-iteration dispatch + readback, 6×6 solve, pose compose. |
| `src/shader/icp_iterate.comp.glsl` (Create) | One ICP iteration on the GPU: transform → grid-NN → per-workgroup `H`,`b` partial reduction. |
| `src/Engine/Pipeline/Registration/Tracker.cpp` (Modify) | Add `GpuIcpTracker` (owns a lazily-created `Engine::Core::Context`, wraps `GpuPointToPlaneIcp`, crops target to source AABB). Register `"icp"` → GPU, `"icp-cpu"` → the existing CPU `PointToPlaneIcpTracker`. |
| `test/test_gpuIcp.cpp` (Create) | Unit tests: `LocalGrid` NN == brute force; GPU one-iteration `H`,`b` ≈ CPU reference; GPU `Solve` pose ≈ CPU `AlignPointToPlaneIcp` on the corner fixture. |

---

## Task 1: `LocalGrid` — CPU crop + uniform-grid NN (no GPU yet)

**Files:**
- Create: `src/Engine/Pipeline/Registration/GpuIcp.h`
- Create: `src/Engine/Pipeline/Registration/GpuIcp.cpp`
- Test: `test/test_gpuIcp.cpp`

**Interfaces:**
- Produces: `Engine::Pipeline::LocalGrid` with:
  - `LocalGrid(const std::vector<Eigen::Vector3f>& pts, float cell)` — buckets `pts` into a dense uniform grid over their AABB.
  - `int Nearest(const Eigen::Vector3f& q, float radius) const` — index of nearest `pts` within `radius`, else `-1`.
  - Public flat arrays for GPU upload: `Eigen::Vector3f m_origin; Eigen::Vector3i m_dims; float m_cell; std::vector<uint32_t> m_bucketStart; /*size dims.x*dims.y*dims.z+1*/ std::vector<uint32_t> m_bucketIdx; /*size pts.size(), pts indices grouped by cell*/`.

- [ ] **Step 1: Write the failing test**

```cpp
// test/test_gpuIcp.cpp
#include "Engine/Pipeline/Registration/GpuIcp.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <random>
#include <vector>
using Engine::Pipeline::LocalGrid;
using Eigen::Vector3f;

// Brute-force nearest within radius (reference).
static int bruteNearest(const std::vector<Vector3f>& pts, const Vector3f& q, float radius) {
    int best = -1; float bestD2 = radius * radius;
    for (int i = 0; i < (int)pts.size(); ++i) {
        const float d2 = (pts[i] - q).squaredNorm();
        if (d2 < bestD2) { bestD2 = d2; best = i; }
    }
    return best;
}

TEST(LocalGrid, NearestMatchesBruteForce) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Vector3f> pts;
    for (int i = 0; i < 2000; ++i) pts.emplace_back(u(rng), u(rng), u(rng));
    const float cell = 0.1f, radius = 0.1f;
    LocalGrid grid(pts, cell);
    for (int i = 0; i < 500; ++i) {
        const Vector3f q(u(rng), u(rng), u(rng));
        const int g = grid.Nearest(q, radius);
        const int b = bruteNearest(pts, q, radius);
        if (b < 0) { EXPECT_LT(g, 0); continue; }
        // Grid and brute may pick different indices only if equidistant; compare distances.
        ASSERT_GE(g, 0);
        EXPECT_NEAR((pts[g] - q).norm(), (pts[b] - q).norm(), 1e-5f);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|GpuIcp.h"`
Expected: FAIL to compile — `GpuIcp.h` not found / `LocalGrid` undefined.

- [ ] **Step 3: Write minimal implementation**

```cpp
// src/Engine/Pipeline/Registration/GpuIcp.h
#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <vector>

namespace Engine::Pipeline {

    // Dense uniform grid over a point set's AABB (cell = correspondence radius). Buckets are stored as
    // a CSR-style pair (bucketStart prefix-sum + bucketIdx grouped indices) so the SAME arrays upload
    // straight to the GPU. Built once per ICP solve over the CROPPED (local) target -> small + cheap.
    class LocalGrid {
    public:
        LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell);
        int Nearest(const Eigen::Vector3f &q, float radius) const;

        Eigen::Vector3f m_origin;                // AABB min
        Eigen::Vector3i m_dims{1, 1, 1};         // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;     // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;       // size pts.size() (pt indices grouped by cell)
        const std::vector<Eigen::Vector3f> &m_pts;

    private:
        int cellIndex(const Eigen::Vector3i &c) const {
            return (c.z() * m_dims.y() + c.y()) * m_dims.x() + c.x();
        }
        Eigen::Vector3i cellOf(const Eigen::Vector3f &p) const {
            return Eigen::Vector3i(int(std::floor((p.x() - m_origin.x()) / m_cell)),
                                   int(std::floor((p.y() - m_origin.y()) / m_cell)),
                                   int(std::floor((p.z() - m_origin.z()) / m_cell)));
        }
    };

} // namespace Engine::Pipeline
```

```cpp
// src/Engine/Pipeline/Registration/GpuIcp.cpp
#include "Engine/Pipeline/Registration/GpuIcp.h"
#include <algorithm>
#include <cmath>

namespace Engine::Pipeline {

    LocalGrid::LocalGrid(const std::vector<Eigen::Vector3f> &pts, float cell) : m_pts(pts) {
        m_cell = cell > 1e-8f ? cell : 1e-8f;
        if (pts.empty()) { m_bucketStart.assign(2, 0); return; }
        Eigen::Vector3f mn = pts[0], mx = pts[0];
        for (const auto &p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
        m_origin = mn;
        for (int a = 0; a < 3; ++a)
            m_dims[a] = std::max(1, int(std::floor((mx[a] - mn[a]) / m_cell)) + 1);
        const int nCells = m_dims.x() * m_dims.y() * m_dims.z();

        // Counting sort of point indices by cell -> CSR (bucketStart prefix sum, bucketIdx grouped).
        m_bucketStart.assign(nCells + 1, 0);
        for (const auto &p : pts) ++m_bucketStart[cellIndex(cellOf(p)) + 1];
        for (int i = 0; i < nCells; ++i) m_bucketStart[i + 1] += m_bucketStart[i];
        m_bucketIdx.resize(pts.size());
        std::vector<uint32_t> cursor(m_bucketStart.begin(), m_bucketStart.end() - 1);
        for (int i = 0; i < (int)pts.size(); ++i)
            m_bucketIdx[cursor[cellIndex(cellOf(pts[i]))]++] = uint32_t(i);
    }

    int LocalGrid::Nearest(const Eigen::Vector3f &q, float radius) const {
        if (m_pts.empty()) return -1;
        const Eigen::Vector3i c = cellOf(q);
        const float r2 = radius * radius;
        int best = -1; float bestD2 = r2;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const Eigen::Vector3i cc(c.x() + dx, c.y() + dy, c.z() + dz);
                    if ((cc.array() < 0).any() || (cc.array() >= m_dims.array()).any()) continue;
                    const int ci = cellIndex(cc);
                    for (uint32_t k = m_bucketStart[ci]; k < m_bucketStart[ci + 1]; ++k) {
                        const int idx = int(m_bucketIdx[k]);
                        const float d2 = (q - m_pts[idx]).squaredNorm();
                        if (d2 < bestD2) { bestD2 = d2; best = idx; }
                    }
                }
        return best;
    }

} // namespace Engine::Pipeline
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|Built target vkspatial_tests$" && ./build/test/vkspatial_tests --gtest_filter='LocalGrid.*'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Pipeline/Registration/GpuIcp.h src/Engine/Pipeline/Registration/GpuIcp.cpp test/test_gpuIcp.cpp
git commit -m "feat(icp): LocalGrid CPU uniform-grid NN (GPU-upload-ready CSR)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `icp_iterate.comp.glsl` + one-iteration GPU accumulation

**Files:**
- Create: `src/shader/icp_iterate.comp.glsl`
- Modify: `src/Engine/Pipeline/Registration/GpuIcp.h`, `src/Engine/Pipeline/Registration/GpuIcp.cpp`
- Test: `test/test_gpuIcp.cpp`

**Interfaces:**
- Consumes: `LocalGrid` (Task 1).
- Produces: `class GpuPointToPlaneIcp { GpuPointToPlaneIcp(Engine::Core::Context&); IterOut Accumulate(const std::vector<Eigen::Vector3f>& src, const Engine::Registration::PointCloud& tgt, const Eigen::Matrix4f& T, float maxCorrDist); };` where `struct IterOut { Eigen::Matrix<double,6,6> H; Eigen::Matrix<double,6,1> b; int inliers; };`. `Accumulate` runs ONE GPU dispatch and returns the (un-centred-frame) normal equations. Internally centres on the target centroid and un-centres the returned `H`,`b` back to `T`'s frame is NOT needed here — Accumulate works entirely in T's frame using centred coordinates only for fixed-point safety (subtract centroid from src', tgt', and adjust the residual is invariant to a common translation, so H,b are identical). Document that H,b equal the CPU reference on the SAME points.

**Design notes for the shader (put as a comment header in the file):**
- One thread per source point. `p = (T * vec4(src_i - c, 1)).xyz` where `c` = target centroid (uniform); target points are pre-shifted by `-c` on the CPU before upload. A common translation `-c` applied to BOTH `p` and `q` leaves the residual `(p-q)·n` and Jacobian `J=[p×n, n]`… note `p×n` is NOT translation-invariant, so **also shift by -c inside the cross product consistently**: the plan centres EVERYTHING on `c`, and the recovered pose is un-centred on the CPU (Task 3). Within one Accumulate call all math is in the centred frame, which is what the CPU reference also uses — so the test compares centred-frame `H`,`b`.
- Reduction: each thread adds its 28 contributions (21 upper-triangular `H` + 6 `b` + 1 inlier) into `shared int acc[28]` via `atomicAdd` (fixed-point ×10000; centred magnitudes keep a 256-point workgroup sum well within int32). `barrier();` then thread 0 writes `acc[28]` to `partials[gl_WorkGroupID.x * 28 + k]`. No cross-workgroup atomics.

- [ ] **Step 1: Write the failing test**

```cpp
// append to test/test_gpuIcp.cpp
#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuIcp.h"
#include "Engine/Registration/RegistrationTypes.h"

// CPU reference: point-to-plane H,b in T's frame, over grid-NN correspondences, CENTRED on tgt centroid.
static void cpuAccumulate(const std::vector<Vector3f>& src, const Engine::Registration::PointCloud& tgt,
                          const Eigen::Matrix4f& T, float maxCorr,
                          Eigen::Matrix<double,6,6>& H, Eigen::Matrix<double,6,1>& b, int& inliers) {
    Vector3f c = Vector3f::Zero();
    for (auto& q : tgt.points) c += q; c /= float(std::max<size_t>(1, tgt.points.size()));
    std::vector<Vector3f> tc(tgt.points.size());
    for (size_t i=0;i<tc.size();++i) tc[i] = tgt.points[i] - c;
    LocalGrid grid(tc, maxCorr);
    H.setZero(); b.setZero(); inliers = 0;
    const Eigen::Matrix3f R = T.block<3,3>(0,0); const Vector3f t = T.block<3,1>(0,3);
    for (auto& s : src) {
        const Vector3f p = R * (s - c) + t;
        const int qi = grid.Nearest(p, maxCorr);
        if (qi < 0) continue;
        const Vector3f& q = tc[qi]; const Vector3f& n = tgt.normals[qi];
        const float e = (p - q).dot(n);
        Eigen::Matrix<float,6,1> J; J.head<3>() = p.cross(n); J.tail<3>() = n;
        H += (J * J.transpose()).cast<double>(); b += (-J * e).cast<double>(); ++inliers;
    }
}

TEST(GpuIcp, AccumulateMatchesCpu) {
    Engine::Core::Context ctx;
    // A small +Z plane patch as source; a matching plane as target (with +Z normals).
    std::vector<Vector3f> src; Engine::Registration::PointCloud tgt;
    for (int i=-15;i<=15;++i) for (int j=-15;j<=15;++j) {
        src.emplace_back(i*0.02f, j*0.02f, 0.01f);          // 1 cm above the target plane
        tgt.points.emplace_back(i*0.02f, j*0.02f, 0.0f);
        tgt.normals.emplace_back(0,0,1);
    }
    const Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float maxCorr = 0.05f;

    Eigen::Matrix<double,6,6> Hc; Eigen::Matrix<double,6,1> bc; int nc;
    cpuAccumulate(src, tgt, T, maxCorr, Hc, bc, nc);
    ASSERT_GT(nc, 100);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto out = gpu.Accumulate(src, tgt, T, maxCorr);
    EXPECT_EQ(out.inliers, nc);
    EXPECT_TRUE(((out.H - Hc).array().abs() < 1e-2 * (1.0 + Hc.array().abs())).all()) << out.H << "\n---\n" << Hc;
    EXPECT_TRUE(((out.b - bc).array().abs() < 1e-2 * (1.0 + bc.array().abs())).all()) << out.b << "\n---\n" << bc;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|GpuPointToPlaneIcp"`
Expected: FAIL — `GpuPointToPlaneIcp` undefined.

- [ ] **Step 3: Write minimal implementation**

Add to `GpuIcp.h`:

```cpp
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Registration/RegistrationTypes.h"
#include <Eigen/Dense>
#include <memory>

namespace Engine::Pipeline {
    class GpuPointToPlaneIcp {
    public:
        struct IterOut { Eigen::Matrix<double, 6, 6> H; Eigen::Matrix<double, 6, 1> b; int inliers; };
        explicit GpuPointToPlaneIcp(Engine::Core::Context &ctx);
        IterOut Accumulate(const std::vector<Eigen::Vector3f> &src,
                           const Engine::Registration::PointCloud &tgt,
                           const Eigen::Matrix4f &T, float maxCorrDist);
    private:
        static constexpr uint32_t kLocal = 256;
        static constexpr float kScale = 10000.0f;
        Engine::Core::Context *m_ctx;
        std::unique_ptr<Engine::Core::ComputePipeline> m_kernel;
        std::unique_ptr<Engine::Core::Buffer> m_src, m_tgtPts, m_tgtNrm, m_bucketStart, m_bucketIdx, m_partials;
    };
}
```

Create `src/shader/icp_iterate.comp.glsl`:

```glsl
#version 450
/// One point-to-plane ICP iteration. Thread = source point. Transforms by push-constant T (centred
/// frame), grid-NN correspondence, then reduces the 6x6 normal equations H,b per-workgroup into shared
/// fixed-point ints (no MoltenVK float atomics); thread 0 writes this workgroup's 28-int partial.
layout(local_size_x = 256) in;
const float SCALE = 10000.0;

// SCALAR push-constant fields ONLY (no vec3/ivec3): GLSL aligns vec3 to 16 bytes, which would NOT
// match the tightly-packed C++ IcpPC struct. Reconstruct vectors in main().
layout(push_constant) uniform PC {
    mat4  g_T;          // current pose (centred frame), column-major
    float g_originX, g_originY, g_originZ, g_cell;  // grid AABB min + cell size
    int   g_dimsX, g_dimsY, g_dimsZ;                // cells per axis
    float g_maxCorr;
    uint  g_numSrc, g_numCells;
};
layout(std430, binding=0) readonly buffer Src        { vec4 g_src[]; };        // xyz used (centred)
layout(std430, binding=1) readonly buffer TgtPts     { vec4 g_tgtPts[]; };     // centred
layout(std430, binding=2) readonly buffer TgtNrm     { vec4 g_tgtNrm[]; };
layout(std430, binding=3) readonly buffer BucketStart{ uint g_bstart[]; };
layout(std430, binding=4) readonly buffer BucketIdx  { uint g_bidx[]; };
layout(std430, binding=5) buffer Partials            { int g_part[]; };        // [numWG * 28]

shared int s_acc[28];

int cellIndex(ivec3 c, ivec3 dims) { return (c.z * dims.y + c.y) * dims.x + c.x; }

void main() {
    vec3  g_origin = vec3(g_originX, g_originY, g_originZ);
    ivec3 g_dims   = ivec3(g_dimsX, g_dimsY, g_dimsZ);

    uint tid = gl_LocalInvocationID.x;
    if (tid < 28u) s_acc[tid] = 0;
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
            float e = dot(p - q, n);
            float J[6];
            vec3 pxn = cross(p, n);
            J[0]=pxn.x; J[1]=pxn.y; J[2]=pxn.z; J[3]=n.x; J[4]=n.y; J[5]=n.z;
            int k = 0;                                   // upper-triangular H (row-major, 21 entries)
            for (int r=0; r<6; ++r) for (int col=r; col<6; ++col)
                atomicAdd(s_acc[k++], int(round(J[r]*J[col]*SCALE)));
            for (int r=0; r<6; ++r)
                atomicAdd(s_acc[21+r], int(round(-J[r]*e*SCALE)));
            atomicAdd(s_acc[27], 1);
        }
    }
    barrier();
    if (tid < 28u) g_part[gl_WorkGroupID.x * 28u + tid] = s_acc[tid];
}
```

Add to `GpuIcp.cpp`:

```cpp
#include <cmath>

namespace Engine::Pipeline {
    namespace {
        struct IcpPC {
            float T[16];               // column-major mat4
            float originX, originY, originZ, cell;
            int32_t dimsX, dimsY, dimsZ; float maxCorr;
            uint32_t numSrc, numCells;
        };
        void writeVec3Buf(Engine::Core::Buffer &buf, const std::vector<Eigen::Vector3f> &v) {
            std::vector<float> pad(v.size() * 4u, 0.0f);           // vec4 std430 stride
            for (size_t i = 0; i < v.size(); ++i) { pad[i*4]=v[i].x(); pad[i*4+1]=v[i].y(); pad[i*4+2]=v[i].z(); }
            buf.Allocate(pad.size() * sizeof(float));
            buf.Upload(pad.data(), pad.size() * sizeof(float));
        }
    }

    GpuPointToPlaneIcp::GpuPointToPlaneIcp(Engine::Core::Context &ctx) : m_ctx(&ctx) {
        m_src = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtPts = std::make_unique<Engine::Core::Buffer>(ctx);
        m_tgtNrm = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketStart = std::make_unique<Engine::Core::Buffer>(ctx);
        m_bucketIdx = std::make_unique<Engine::Core::Buffer>(ctx);
        m_partials = std::make_unique<Engine::Core::Buffer>(ctx);
        m_kernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_kernel->Build("icp_iterate.comp.glsl");
    }

    GpuPointToPlaneIcp::IterOut GpuPointToPlaneIcp::Accumulate(
            const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
            const Eigen::Matrix4f &T, float maxCorrDist) {
        IterOut out; out.H.setZero(); out.b.setZero(); out.inliers = 0;
        if (src.empty() || tgt.points.size() < 3) return out;

        // Centre on the target centroid (numerical conditioning + fixed-point safety).
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        for (const auto &q : tgt.points) c += q; c /= float(tgt.points.size());
        std::vector<Eigen::Vector3f> sc(src.size()), tc(tgt.points.size());
        for (size_t i = 0; i < src.size(); ++i) sc[i] = src[i] - c;
        for (size_t i = 0; i < tc.size(); ++i) tc[i] = tgt.points[i] - c;

        LocalGrid grid(tc, maxCorrDist);
        writeVec3Buf(*m_src, sc);
        writeVec3Buf(*m_tgtPts, tc);
        writeVec3Buf(*m_tgtNrm, tgt.normals);
        m_bucketStart->Allocate(grid.m_bucketStart.size() * sizeof(uint32_t));
        m_bucketStart->Upload(grid.m_bucketStart.data(), grid.m_bucketStart.size() * sizeof(uint32_t));
        m_bucketIdx->Allocate(std::max<size_t>(1, grid.m_bucketIdx.size()) * sizeof(uint32_t));
        if (!grid.m_bucketIdx.empty())
            m_bucketIdx->Upload(grid.m_bucketIdx.data(), grid.m_bucketIdx.size() * sizeof(uint32_t));

        const uint32_t numWG = (uint32_t(src.size()) + kLocal - 1) / kLocal;
        m_partials->AllocateHostVisibleReadback(numWG * 28u * sizeof(int32_t));
        std::memset(m_partials->MappedPtr(), 0, numWG * 28u * sizeof(int32_t));
        m_partials->FlushMapped(numWG * 28u * sizeof(int32_t));

        IcpPC pc{};
        for (int i = 0; i < 16; ++i) pc.T[i] = T.data()[i]; // Eigen is column-major -> matches std430 mat4
        pc.originX = grid.m_origin.x(); pc.originY = grid.m_origin.y(); pc.originZ = grid.m_origin.z();
        pc.cell = grid.m_cell; pc.dimsX = grid.m_dims.x(); pc.dimsY = grid.m_dims.y(); pc.dimsZ = grid.m_dims.z();
        pc.maxCorr = maxCorrDist; pc.numSrc = uint32_t(src.size());
        pc.numCells = uint32_t(grid.m_dims.x() * grid.m_dims.y() * grid.m_dims.z());

        m_kernel->Bind(0, *m_src).Bind(1, *m_tgtPts).Bind(2, *m_tgtNrm)
                 .Bind(3, *m_bucketStart).Bind(4, *m_bucketIdx).Bind(5, *m_partials);
        m_kernel->Args(pc);
        m_kernel->DispatchElements(uint32_t(src.size())); // synchronous

        m_partials->InvalidateMapped(numWG * 28u * sizeof(int32_t));
        const int32_t *part = static_cast<const int32_t *>(m_partials->MappedPtr());
        double acc[28] = {0};
        for (uint32_t w = 0; w < numWG; ++w) for (int k = 0; k < 28; ++k) acc[k] += part[w * 28u + k];
        int k = 0;
        for (int r = 0; r < 6; ++r) for (int col = r; col < 6; ++col) {
            const double v = acc[k++] / double(kScale);
            out.H(r, col) = v; out.H(col, r) = v;
        }
        for (int r = 0; r < 6; ++r) out.b(r) = acc[21 + r] / double(kScale);
        out.inliers = int(std::llround(acc[27])); // inlier count stored ×1 (SCALE not applied to it)
        return out;
    }
}
```

> Note: index 27 (inlier count) is accumulated as a raw `+1` in the shader (no `SCALE`), so the readback reads it directly (`int(acc[27])`) — only the 27 `H`/`b` entries are divided by `kScale`. The code above already does this correctly.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake . -B build >/dev/null && cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|Built target vkspatial_tests$" && ./build/test/vkspatial_tests --gtest_filter='GpuIcp.AccumulateMatchesCpu'`
Expected: PASS (GPU `H`,`b`,inliers ≈ CPU reference).

- [ ] **Step 5: Commit**

```bash
git add src/shader/icp_iterate.comp.glsl src/Engine/Pipeline/Registration/GpuIcp.h src/Engine/Pipeline/Registration/GpuIcp.cpp test/test_gpuIcp.cpp
git commit -m "feat(icp): GPU one-iteration point-to-plane H,b accumulation (per-workgroup fixed-point reduction)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `GpuPointToPlaneIcp::Solve` — full iterate loop + 6×6 solve + un-centre

**Files:**
- Modify: `src/Engine/Pipeline/Registration/GpuIcp.h`, `src/Engine/Pipeline/Registration/GpuIcp.cpp`
- Test: `test/test_gpuIcp.cpp`

**Interfaces:**
- Produces: `Engine::Registration::RegistrationResult GpuPointToPlaneIcp::Solve(const std::vector<Eigen::Vector3f>& src, const Engine::Registration::PointCloud& tgt, const Eigen::Matrix4f& priorT, const Engine::Registration::RegistrationParam& params);` — result-equivalent to `Engine::Registration::AlignPointToPlaneIcp`.

**Design — work entirely in the centred frame, un-centre once at the end.** Everything (source, target, pose, Jacobian `p×n`) is expressed relative to the target centroid `c`, so all magnitudes stay small (fixed-point safe) and the math is self-consistent. Seed `T_centred = Tc · priorT · Tc⁻¹` (`Tc = Translate(-c)`). Each iteration: `AccumulateCentred` returns centred-frame `H`,`b`; solve `x = H.ldlt().solve(b)`; compose the incremental twist onto `T` exactly as the CPU code does (`AngleAxis Z*Y*X`, `delta*T`); break on `inliers < minInliers` or `x.norm() < convEps`. After the loop, un-centre: `res.T = Tc⁻¹ · T_centred · Tc`. This is the same recovered world pose as the CPU `AlignPointToPlaneIcp` (whose points happen to be near the origin already), so the Task-3 test compares the two world poses directly.

- [ ] **Step 1: Write the failing test**

```cpp
// append to test/test_gpuIcp.cpp
#include "Engine/Registration/Icp.h"
#include <Eigen/Geometry>

TEST(GpuIcp, SolveMatchesCpuOnCorner) {
    Engine::Core::Context ctx;
    // A 3-plane corner target (constrains all 6 DoF); source = target perturbed by a small transform.
    Engine::Registration::PointCloud tgt; std::vector<Vector3f> src;
    auto addPlane = [&](const Vector3f& o, const Vector3f& u, const Vector3f& v, const Vector3f& n){
        for (int i=-10;i<=10;++i) for (int j=-10;j<=10;++j) {
            const Vector3f p = o + u*(i*0.03f) + v*(j*0.03f);
            tgt.points.push_back(p); tgt.normals.push_back(n);
        }
    };
    addPlane({0,0,0},{1,0,0},{0,1,0},{0,0,1});
    addPlane({0,0,0},{0,1,0},{0,0,1},{1,0,0});
    addPlane({0,0,0},{1,0,0},{0,0,1},{0,1,0});
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    for (const auto& q : tgt.points) src.push_back(perturb * q); // source is the model, moved

    Engine::Registration::RegistrationParam params; params.maxCorrDist = 0.1f; params.maxIters = 30;
    const auto cpu = Engine::Registration::AlignPointToPlaneIcp(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(cpu.valid);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto g = gpu.Solve(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(g.valid);
    // Both should recover ~perturb⁻¹ (align source back onto target). Compare the two poses directly.
    EXPECT_TRUE(((g.T - cpu.T).array().abs() < 5e-3f).all()) << "gpu:\n" << g.T << "\ncpu:\n" << cpu.T;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|no member named 'Solve'"`
Expected: FAIL — `Solve` undefined.

- [ ] **Step 3: Write minimal implementation**

Add `Solve` declaration to `GpuIcp.h` (next to `Accumulate`) and implement in `GpuIcp.cpp`:

```cpp
Engine::Registration::RegistrationResult GpuPointToPlaneIcp::Solve(
        const std::vector<Eigen::Vector3f> &src, const Engine::Registration::PointCloud &tgt,
        const Eigen::Matrix4f &priorT, const Engine::Registration::RegistrationParam &params) {
    Engine::Registration::RegistrationResult res; res.T = priorT;
    if (src.empty() || tgt.points.size() < 3 || tgt.normals.size() != tgt.points.size()) return res;

    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto &q : tgt.points) c += q; c /= float(tgt.points.size());
    Eigen::Matrix4f Tc = Eigen::Matrix4f::Identity(); Tc.block<3,1>(0,3) = -c;  // shift world->centred
    Eigen::Matrix4f TcInv = Eigen::Matrix4f::Identity(); TcInv.block<3,1>(0,3) = c;
    Eigen::Matrix4f T = Tc * priorT * TcInv; // work in the centred frame

    for (int iter = 0; iter < params.maxIters; ++iter) {
        const IterOut a = AccumulateCentred(src, tgt, c, T, params.maxCorrDist); // (see below)
        if (a.inliers < params.minInliers) break;
        const Eigen::Matrix<double,6,1> x = a.H.ldlt().solve(a.b);
        const Eigen::Matrix3d Rd = (Eigen::AngleAxisd(x[2], Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(x[1], Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(x[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = Rd.cast<float>(); delta.block<3,1>(0,3) = x.tail<3>().cast<float>();
        T = delta * T;
        res.numInliers = size_t(a.inliers);
        res.fitness = float(a.inliers) / float(src.size());
        if (x.norm() < params.convEps) break;
    }
    res.T = TcInv * T * Tc;                    // un-centre back to world
    res.valid = res.numInliers >= size_t(params.minInliers);
    return res;
}
```

Refactor Task 2's `Accumulate` into `AccumulateCentred(src, tgt, c, T, maxCorr)` that takes the precomputed centroid `c` and the CENTRED working pose `T` (so it does not re-centre `T`). It builds `sc = src - c`, `tc = tgt.points - c`, uploads, dispatches with `T`, and returns the centred-frame `H`,`b`. Keep the public `Accumulate` (Task 2 test) as a thin wrapper that computes `c` from `tgt` and calls `AccumulateCentred(src, tgt, c, T, maxCorr)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|Built target vkspatial_tests$" && ./build/test/vkspatial_tests --gtest_filter='GpuIcp.*'`
Expected: PASS (GPU pose within 5e-3 of CPU on the corner fixture).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Pipeline/Registration/GpuIcp.h src/Engine/Pipeline/Registration/GpuIcp.cpp test/test_gpuIcp.cpp
git commit -m "feat(icp): GpuPointToPlaneIcp::Solve (centred iterate loop + LDLT + un-centre)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `GpuIcpTracker` + register as `"icp"` (CPU → `"icp-cpu"`)

**Files:**
- Modify: `src/Engine/Pipeline/Registration/Tracker.cpp`
- Test: `test/test_pipeline.cpp` (add a `--tracker icp` end-to-end smoke)

**Interfaces:**
- Consumes: `GpuPointToPlaneIcp::Solve` (Task 3), `TrackerRegistry`, `Tracker`, `TrackingResult`.
- Produces: registry name `"icp"` → GPU tracker, `"icp-cpu"` → the existing CPU `PointToPlaneIcpTracker`.

**Design:** `GpuIcpTracker` owns `std::unique_ptr<Engine::Core::Context> m_ctx` and `std::unique_ptr<GpuPointToPlaneIcp> m_gpu`, both created lazily on the FIRST `Track` (which runs on the ICP thread — matches how `IntegrationThread` creates its context inside `Run`). `Track` builds `tgt` from the model entries **cropped to the source AABB + `maxCorrDist` margin** (so the upload + grid are local, not O(full model)), then calls `m_gpu->Solve`.

- [ ] **Step 1: Write the failing test**

```cpp
// append to test/test_pipeline.cpp
TEST(Pipeline, GpuIcpTrackerRuns) {
    const FrameDir frames(3);
    ep::Pipeline::Config cfg = makeConfig(frames.files, 0.0);
    ep::Pipeline pipe(std::move(cfg), ep::TrackerRegistry::Default().Create("icp"));
    ASSERT_NE(ep::TrackerRegistry::Default().Create("icp"), nullptr);
    ASSERT_NE(ep::TrackerRegistry::Default().Create("icp-cpu"), nullptr);
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 2)) << "gpu-icp pipeline did not integrate frames";
    pipe.CheckErrors();
    EXPECT_NE(pipe.LatestModel(), nullptr);
    pipe.Stop();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|icp-cpu"`
Expected: FAIL — `Create("icp-cpu")` returns null (unregistered) / assertion fails.

- [ ] **Step 3: Write minimal implementation**

In `Tracker.cpp`, add (inside the anonymous namespace) a `GpuIcpTracker`, and change `Default()`:

```cpp
#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuIcp.h"

class GpuIcpTracker : public Tracker {
public:
    const char *Name() const override { return "icp"; }
    TrackingResult Track(const Frame &frame, const ModelSnapshot *model,
                         const Eigen::Isometry3f &priorPose) override {
        TrackingResult r; r.pose = priorPose;
        if (model == nullptr || model->entries.empty() || frame.pts.empty()) return r;
        if (!m_ctx) { m_ctx = std::make_unique<Engine::Core::Context>();
                      m_gpu = std::make_unique<GpuPointToPlaneIcp>(*m_ctx); } // lazy, on the ICP thread

        // Crop the model to the source AABB + margin so upload/grid stay local (not O(full model)).
        Eigen::Vector3f mn = frame.pts[0], mx = frame.pts[0];
        for (const auto &p : frame.pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
        const float m = m_params.maxCorrDist;
        mn.array() -= m; mx.array() += m;
        Engine::Registration::PointCloud tgt;
        tgt.points.reserve(model->entries.size()); tgt.normals.reserve(model->entries.size());
        for (const auto &e : model->entries)
            if ((e.center.array() >= mn.array()).all() && (e.center.array() <= mx.array()).all()) {
                tgt.points.push_back(e.center); tgt.normals.push_back(e.normal);
            }
        if (tgt.points.size() < 3) return r; // nothing local to align to -> keep prior

        const auto icp = m_gpu->Solve(frame.pts, tgt, priorPose.matrix(), m_params);
        r.pose = Eigen::Isometry3f(icp.T); r.fitness = icp.fitness;
        r.inliers = icp.numInliers; r.valid = icp.valid;
        return r;
    }
private:
    Engine::Registration::RegistrationParam m_params;
    std::unique_ptr<Engine::Core::Context> m_ctx;
    std::unique_ptr<GpuPointToPlaneIcp> m_gpu;
};
```

```cpp
TrackerRegistry TrackerRegistry::Default() {
    TrackerRegistry reg;
    reg.Register("identity", [] { return std::make_unique<IdentityTracker>(); });
    reg.Register("icp", [] { return std::make_unique<GpuIcpTracker>(); });
    reg.Register("icp-cpu", [] { return std::make_unique<PointToPlaneIcpTracker>(); });
    reg.Register("global", [] { return std::make_unique<GlobalRegistrationTracker>(); });
    return reg;
}
```

> The default `RegistrationParam.maxCorrDist = 0.1` is too tight for `voxel 0.5` maps; set the tracker's `m_params.maxCorrDist` to `2×voxel` if a per-tracker voxel becomes available. Out of scope here — note as a follow-up.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --target vkspatial_tests voxel_fill_debugger -j8 2>&1 | grep -E "error:|Built target" && ./build/test/vkspatial_tests --gtest_filter='Pipeline.GpuIcpTrackerRuns:GpuIcp.*'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Pipeline/Registration/Tracker.cpp test/test_pipeline.cpp
git commit -m "feat(icp): register GPU icp tracker as 'icp' (CPU -> 'icp-cpu'), crop target to source AABB

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Measure GPU vs CPU (RelWithDebInfo) + record

**Files:** none (measurement + notes).

- [ ] **Step 1: Build both trackers in release**

Run: `cmake . -B build-rel >/dev/null && cmake --build build-rel --target voxel_fill_debugger -j8 2>&1 | grep -E "error:|Built target voxel_fill_debugger$"`
Expected: builds.

- [ ] **Step 2: Run the live viewer with each tracker on a real scan and read the ICP stage average**

Run (GUI — human runs it): `./build-rel/example2/voxel_fill_debugger --dir scan_out --voxel 0.5 --tracker icp` then `--tracker icp-cpu`; press Play, read the **ICP** row in the "Pipeline stages (avg ms / frames)" panel for each.
Expected: `icp` (GPU) ICP-stage avg materially below `icp-cpu`, and roughly flat as the map grows (the crop keeps it local).

- [ ] **Step 3: Record the numbers in the plan's results note + decide on Phase 2**

If the GPU per-iteration readback (one `vkQueueWaitIdle` per iteration × `maxIters`) dominates, add a follow-up task: fuse all iterations into one dispatch is impossible (T depends on the previous solve), so instead cap `maxIters` lower with a motion-model prior, or move the 6×6 solve to the GPU to avoid per-iter readback. Record which.

- [ ] **Step 4: Commit the results note**

```bash
git add docs/superpowers/plans/2026-08-06-gpu-icp-tracker.md
git commit -m "docs(icp): GPU vs CPU ICP measurements + Phase 2 decision

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Results (measured, RelWithDebInfo)

**Machine:** Apple M4 Max, MoltenVK (`[Engine::Core::Context] Device: Apple M4 Max`), macOS. Build: `build-rel/` (`CMAKE_BUILD_TYPE=RelWithDebInfo`).

**Method:** Headless deterministic benchmark, `test/test_gpuIcp.cpp` → `TEST(GpuIcp, DISABLED_BenchmarkVsCpu)`. Dense 3-plane corner target (apex off-origin at `(0.3,0.3,0.3)`, normals along +X/+Y/+Z so all 6 DoF are constrained), extent held ~0.6m while point spacing shrinks as N grows (N ≈ 5,043 / 20,667 / 81,675 / 201,243). Source = target transformed by a fixed known perturbation (translate `(0.02,-0.015,0.01)`, rotate 0.03 rad about Z). Both trackers run `Solve(src, tgt, Identity, params)` with the same `RegistrationParam{maxCorrDist=0.03, maxIters=20}` (matched to the worst-case initial misalignment ≈3–4cm so correspondences bootstrap at every density; ratio to point spacing ranges 2×–13× across the four sizes). One warmup call each (excludes GPU shader compile + first-size buffer allocation), then mean of 10 timed repeats via `std::chrono::steady_clock`. Both poses were asserted to agree within `5e-3` at every size (sanity check passed).

Run: `./build-rel/test/vkspatial_tests --gtest_also_run_disabled_tests --gtest_filter='GpuIcp.DISABLED_BenchmarkVsCpu'`

```
         N |  GPU mean ms |  CPU mean ms |    speedup | GPU inliers
-----------|--------------|--------------|------------|------------
      5043 |        3.441 |        1.420 |      0.41x | 5043
     20667 |        5.411 |        9.934 |      1.84x | 20667
     81675 |       14.043 |      100.957 |      7.19x | 81675
    201243 |       28.907 |      577.185 |     19.97x | 201243
```

(Re-run confirmed run-to-run stable: 0.40x / 1.76x / 7.14x / 21.43x — same shape.)

Note: this measures ONLY the core ICP solve on equal-N target/source clouds. In the real pipeline, `GpuIcpTracker::Track` additionally crops the model to the frame's source AABB (+ `maxCorrDist` margin) before calling `Solve`, which is an *additional* GPU-side advantage (smaller effective N per frame as the map grows) that this microbenchmark does not capture — real-world numbers should look at least as good as this table, not worse.

**Crossover:** between N=5,043 and N=20,667. Below the crossover the GPU tracker is **slower** than CPU (0.4×) — GPU per-iteration overhead dominates when there's little actual compute to hide it behind. Above it, the GPU pulls ahead fast: 1.8× at 20k, 7.2× at 80k, ~20–21× at 200k, and CPU cost is clearly scaling worse than linearly with N (grid-cell occupancy grows with density at fixed `maxCorrDist`) while GPU cost grows much more slowly (28.9ms at 200k vs 3.4ms at 5k — an ~8× time increase for a ~40× increase in N).

### Phase-2 decision

**Root cause identified, and it is worse than "just re-uploading buffers": inspecting `GpuPointToPlaneIcp::Solve`/`AccumulateCentred` (`src/Engine/Pipeline/Registration/GpuIcp.cpp`) shows that every one of the up-to-20 ICP iterations rebuilds the CPU-side `LocalGrid` from scratch (an O(N) counting sort over the target points) and re-uploads all of `src`, `tgt.points`, `tgt.normals`, and the grid's `bucketStart`/`bucketIdx` arrays to the GPU (each via `Buffer::Allocate` + `Upload`) — even though target and source point sets, and therefore the grid, are 100% invariant across the iterations of a single `Solve` call. The only thing that legitimately changes iteration-to-iteration is the push-constant pose `T`. On top of that, `DispatchElements` is synchronous (comment: `// synchronous`, implies a `vkQueueWaitIdle`-style per-dispatch stall) and the partials buffer is realloc'd + `memset` + `FlushMapped` every iteration too. This fully explains the shape of the results: at low N (5k) this fixed per-iteration overhead (CPU grid rebuild + 5 buffer allocate/upload round-trips + a synchronous dispatch) costs more than the CPU reference's entire solve, so GPU loses; at high N the O(N) GPU compute finally amortizes the fixed overhead and wins big.

**Recommendation: prioritize hoisting the per-iteration grid-build and buffer uploads out of the iterate loop.** Concretely, inside `Solve`: build the `LocalGrid` and upload `src`/`tgt.points`/`tgt.normals`/`bucketStart`/`bucketIdx` **once**, before the iteration loop begins (they don't depend on `T`); then each iteration should only update the push-constant `T`, dispatch, and read back `H`,`b`. This directly targets the plan's anticipated follow-up ("eliminate per-iteration readback / stop re-uploading unchanged buffers, since only the push-constant T changes each iteration") but the fix is even more impactful than originally scoped, since it also removes a full O(N) CPU grid rebuild per iteration, not just GPU buffer uploads. Given the measured shape (GPU loses below ~10–15k points purely on fixed overhead), this single change should be enough to flip the small-N case to a GPU win too, making `"icp"` strictly better than `"icp-cpu"` at every map size instead of only above the crossover. Secondary/lower priority (only if the fix above still leaves per-iteration `vkQueueWaitIdle` as the bottleneck): solve the 6×6 on the GPU or reduce `maxIters` via a motion-model prior, to cut the number of CPU↔GPU round trips per `Solve` call.

---

## Follow-ups (out of scope — Phase 2, only if Task 5 shows a need)

- **Persistent model spatial index** (updated incrementally on download) to remove the per-frame O(model) crop scan.
- **Per-iteration readback elimination:** solve the 6×6 on the GPU (or a small fixed-iteration Gauss-Newton fully on GPU) so the CPU is not in the per-iteration loop.
- **maxCorrDist auto-scaled** to the map voxel size (thread the voxel through the tracker config).
- **Workgroup subgroup reduction** (`subgroupAdd`) instead of `shared int` atomics, if profiling shows the reduction is hot.
