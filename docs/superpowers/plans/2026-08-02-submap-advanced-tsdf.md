# SubmapAdvancedTSDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Density-adaptive 2-level detail submaps — a base `TiledAdvancedTSDF` at `baseVoxel` (all points) plus a detail `TiledAdvancedTSDF` at `baseVoxel/2` populated only in dense sub-blocks, for fine detail recovery.

**Architecture:** A header-only `SubmapAdvancedTSDF` composes two `TiledAdvancedTSDF` levels. Batch 2-pass: `AddDensity(all)` → `FinalizeDensity()` (mark blocks whose avg points-per-occupied-base-voxel ≥ k) → `Integrate(all)` (base always; detail only for points in dense blocks) → `ExtractPointCloud()` (detail whole + base with dense-block points dropped = precedence dedup).

**Tech Stack:** C++17, Eigen, existing `TiledAdvancedTSDF` (Vulkan compute), GoogleTest, CMake (GLOB).

## Global Constraints
- Header-only; composes `TiledAdvancedTSDF`; NO changes to existing engine classes. No new third-party deps.
- Detail voxel = `baseVoxel * 0.5`; same truncation (`C+2G≤512` still holds since G scales with trunc/voxel).
- 2 levels only (base + ½). Recursion, online density, and seamless transitional boundaries are out of scope.
- Density: block = `blockVoxels` base-voxels (default 32), `blockWorld = baseVoxel*blockVoxels`; dense if `pointCount / max(1, occupiedBaseVoxels) >= k` (default `k=4`).
- Extraction: precedence dedup — detail kept whole; base point dropped when `blockOf(point)` is dense. Minor coarse↔fine seam accepted.
- Build/test: `VULKAN_SDK=/usr/local`; tests target `vkspatial_tests` with conda-workaround flags `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR`. `git submodule update --init lib/SPIRV-Reflect` in a fresh worktree.
- Git: commit only this feature's files; leave the repo's other uncommitted changes untouched.

---

### Task 1: `SubmapAdvancedTSDF.h` + unit tests

**Files:**
- Create: `src/Engine/Spatial/SubmapAdvancedTSDF.h`
- Create: `test/test_submapAdvancedTsdf.cpp`

**Interfaces:**
- Consumes: `Engine::Spatial::TiledAdvancedTSDF` (`Build(ctx,voxel,trunc,hashCapPerTile,maxPtsPerFrame)`, `SetIntegrationQuality`, `SetPointToPlane`, `SetConfidenceWeight`, `SetHermitePosition`, `Integrate(pts,nrm,cam)`, `ExtractPointCloud(merge)→OrientedPointCloud`, `TileCount()`), `Engine::Spatial::IntegrationQuality`, `Engine::Spatial::OrientedPointCloud` (`.points`,`.normals`), `Engine::Core::Context`.
- Produces: `class Engine::Spatial::SubmapAdvancedTSDF` with the API in the spec (`Build`, `Set*`, `AddDensity`, `FinalizeDensity`, `Integrate`, `ExtractPointCloud`, `DenseBlockCount`, `BaseTileCount`, `DetailTileCount`).

- [ ] **Step 1: Write the failing tests**

Create `test/test_submapAdvancedTsdf.cpp`:

```cpp
#include "Engine/Core/Context.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using Engine::Spatial::OrientedPointCloud;
using Engine::Spatial::SubmapAdvancedTSDF;
using Eigen::Vector3f;

namespace {

    // +Z plane patch of side `span` centered at `c`, points on a grid of pitch `spacing`.
    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, const Vector3f &c,
                   float span, float spacing) {
        const int half = std::max(1, int(span / (2.0f * spacing)));
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(c.x() + i * spacing, c.y() + j * spacing, c.z());
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // Min nearest-neighbour distance over the cloud (brute force; small clouds).
    float minSpacing(const std::vector<Vector3f> &p) {
        float best = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < p.size(); ++i)
            for (std::size_t j = i + 1; j < p.size(); ++j)
                best = std::min(best, (p[i] - p[j]).squaredNorm());
        return std::sqrt(best);
    }

} // namespace

// A locally dense sub-patch triggers a detail submap; extracted points there reach half-voxel
// spacing that a base-only map cannot.
TEST(SubmapAdvanced, DenseRegionGetsDetail) {
    Engine::Core::Context ctx;
    // baseVoxel 0.05, block 32 -> blockWorld 1.6; k=4.
    std::vector<Vector3f> coarseP, coarseN, denseP, denseN;
    makePlane(coarseP, coarseN, Vector3f(6, 6, 6), 8.0f, 0.05f);  // ~1 pt / base voxel -> sparse
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);   // ~4 pt/axis / base voxel -> dense

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.AddDensity(coarseP);
    s.AddDensity(denseP);
    s.FinalizeDensity();
    EXPECT_GT(s.DenseBlockCount(), 0u);
    s.Integrate(coarseP, coarseN, Vector3f(6, 6, 7));
    s.Integrate(denseP, denseN, Vector3f(6, 6, 7));
    EXPECT_GT(s.DetailTileCount(), 0u);

    const OrientedPointCloud cloud = s.ExtractPointCloud(/*merge=*/false);
    ASSERT_GT(cloud.points.size(), 100u);
    // Detail (half of 0.05 = 0.025) present -> some points closer than 0.7*baseVoxel.
    EXPECT_LT(minSpacing(cloud.points), 0.035f);
}

// A uniformly sparse scene triggers no detail; extraction is base-only resolution.
TEST(SubmapAdvanced, SparseSceneNoDetail) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> p, n;
    makePlane(p, n, Vector3f(6, 6, 6), 8.0f, 0.05f); // ~1 pt / base voxel

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.AddDensity(p);
    s.FinalizeDensity();
    EXPECT_EQ(s.DenseBlockCount(), 0u);
    s.Integrate(p, n, Vector3f(6, 6, 7));
    EXPECT_EQ(s.DetailTileCount(), 0u);

    const OrientedPointCloud cloud = s.ExtractPointCloud(/*merge=*/false);
    ASSERT_GT(cloud.points.size(), 100u);
    EXPECT_GT(minSpacing(cloud.points), 0.035f); // base-only ~0.05
}

// Precedence dedup: in dense blocks the base surface is REPLACED by detail, not added on top.
// Detail (half voxel) ~4x base points for a plane; without dedup it would be ~5x (base+detail).
TEST(SubmapAdvanced, BaseReplacedNotAddedInDenseBlocks) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> denseP, denseN;
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f); // one dense patch, dense block

    auto run = [&](float k) {
        SubmapAdvancedTSDF s;
        s.Build(ctx, 0.05f, 0.15f, 32, k);
        s.AddDensity(denseP);
        s.FinalizeDensity();
        s.Integrate(denseP, denseN, Vector3f(6, 6, 7));
        return s.ExtractPointCloud(/*merge=*/false).points.size();
    };
    const double nBase = double(run(1.0e9f)); // k huge -> no dense -> base only
    const double nSub = double(run(4.0f));     // dense -> detail replaces base
    ASSERT_GT(nBase, 50.0);
    EXPECT_GT(nSub, 2.5 * nBase);  // detail present (finer)
    EXPECT_LT(nSub, 4.7 * nBase);  // base was dropped in the dense block (not ~5x = base+detail)
}

// Integrate before FinalizeDensity leaves the detail level empty (usage gate).
TEST(SubmapAdvanced, FinalizeGate) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> denseP, denseN;
    makePlane(denseP, denseN, Vector3f(6, 6, 6), 1.2f, 0.012f);

    SubmapAdvancedTSDF s;
    s.Build(ctx, 0.05f, 0.15f, 32, 4.0f);
    s.Integrate(denseP, denseN, Vector3f(6, 6, 7)); // no FinalizeDensity yet
    EXPECT_EQ(s.DetailTileCount(), 0u);
    EXPECT_EQ(s.DenseBlockCount(), 0u);
}
```

- [ ] **Step 2: Run tests to verify they FAIL (header missing)**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`
Expected: BUILD FAILS — `fatal error: 'Engine/Spatial/SubmapAdvancedTSDF.h' file not found`.

- [ ] **Step 3: Create `SubmapAdvancedTSDF.h`**

Create `src/Engine/Spatial/SubmapAdvancedTSDF.h`:

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Engine::Spatial {

    // Density-adaptive 2-level detail submap: a base TiledAdvancedTSDF at baseVoxel (ALL points)
    // plus a detail TiledAdvancedTSDF at baseVoxel/2 populated only in dense blocks, for detail
    // recovery where points are dense. Batch 2-pass:
    //   AddDensity(all pts) -> FinalizeDensity() -> Integrate(all frames) -> ExtractPointCloud().
    // A block is dense if its avg points-per-occupied-base-voxel >= k (default 4).
    class SubmapAdvancedTSDF {
    public:
        void Build(Engine::Core::Context &ctx, float baseVoxel, float truncation,
                   int blockVoxels = 32, float detailPtsPerVoxel = 4.0f,
                   uint32_t tileHashPerTile = 1u << 20, uint32_t maxPointsPerFrame = 1u << 17) {
            m_baseVoxel = baseVoxel;
            m_blockWorld = baseVoxel * float(blockVoxels);
            m_detailK = detailPtsPerVoxel;
            m_finalized = false;
            m_base.Build(ctx, baseVoxel, truncation, tileHashPerTile, maxPointsPerFrame);
            m_detail.Build(ctx, baseVoxel * 0.5f, truncation, tileHashPerTile, maxPointsPerFrame);
            m_count.clear();
            m_occ.clear();
            m_dense.clear();
        }

        void SetIntegrationQuality(const IntegrationQuality &q) {
            m_base.SetIntegrationQuality(q);
            m_detail.SetIntegrationQuality(q);
        }
        void SetPointToPlane(bool on) {
            m_base.SetPointToPlane(on);
            m_detail.SetPointToPlane(on);
        }
        void SetConfidenceWeight(float lambda) {
            m_base.SetConfidenceWeight(lambda);
            m_detail.SetConfidenceWeight(lambda);
        }
        void SetHermitePosition(bool on) {
            m_base.SetHermitePosition(on);
            m_detail.SetHermitePosition(on);
        }

        // Pass 1: accumulate per-block point count + distinct occupied base-voxel keys.
        void AddDensity(const std::vector<Eigen::Vector3f> &pts) {
            for (const Eigen::Vector3f &p : pts) {
                const BlockKey b = blockOf(p);
                ++m_count[b];
                m_occ[b].insert(voxelKey(p));
            }
        }

        // Mark dense blocks (avg pts / occupied base-voxel >= k); free the accumulators.
        void FinalizeDensity() {
            for (const auto &kv : m_count) {
                auto it = m_occ.find(kv.first);
                const std::size_t occ = (it != m_occ.end() && !it->second.empty()) ? it->second.size() : 1;
                if (float(kv.second) / float(occ) >= m_detailK) m_dense.insert(kv.first);
            }
            m_count.clear();
            m_occ.clear();
            m_finalized = true;
        }

        // Pass 2: base gets all points; detail gets only points whose block is dense.
        void Integrate(const std::vector<Eigen::Vector3f> &pts,
                       const std::vector<Eigen::Vector3f> &nrm,
                       const Eigen::Vector3f &cam = Eigen::Vector3f::Zero()) {
            m_base.Integrate(pts, nrm, cam);
            if (!m_finalized || m_dense.empty()) return;
            std::vector<Eigen::Vector3f> dp, dn;
            const std::size_t n = std::min(pts.size(), nrm.size());
            dp.reserve(n);
            dn.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                if (m_dense.count(blockOf(pts[i]))) {
                    dp.push_back(pts[i]);
                    dn.push_back(nrm[i]);
                }
            if (!dp.empty()) m_detail.Integrate(dp, dn, cam);
        }

        // Detail (whole) + base (points inside dense blocks dropped) -> precedence dedup.
        OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            OrientedPointCloud out = m_detail.ExtractPointCloud(merge);
            const OrientedPointCloud base = m_base.ExtractPointCloud(merge);
            const std::size_t m = std::min(base.points.size(), base.normals.size());
            for (std::size_t i = 0; i < m; ++i) {
                if (m_dense.count(blockOf(base.points[i]))) continue; // detail covers this block
                out.points.push_back(base.points[i]);
                out.normals.push_back(base.normals[i]);
            }
            return out;
        }

        uint32_t DenseBlockCount() const { return static_cast<uint32_t>(m_dense.size()); }
        uint32_t BaseTileCount() const { return m_base.TileCount(); }
        uint32_t DetailTileCount() const { return m_detail.TileCount(); }

    private:
        struct BlockKey {
            int x, y, z;
            bool operator==(const BlockKey &o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct BlockKeyHash {
            std::size_t operator()(const BlockKey &k) const {
                std::size_t h = std::hash<int>()(k.x);
                h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        BlockKey blockOf(const Eigen::Vector3f &p) const {
            return BlockKey{static_cast<int>(std::floor(p.x() / m_blockWorld)),
                            static_cast<int>(std::floor(p.y() / m_blockWorld)),
                            static_cast<int>(std::floor(p.z() / m_blockWorld))};
        }
        int64_t voxelKey(const Eigen::Vector3f &p) const {
            const int64_t vx = static_cast<int64_t>(std::floor(p.x() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vy = static_cast<int64_t>(std::floor(p.y() / m_baseVoxel)) & 0x1FFFFF;
            const int64_t vz = static_cast<int64_t>(std::floor(p.z() / m_baseVoxel)) & 0x1FFFFF;
            return vx | (vy << 21) | (vz << 42);
        }

        float m_baseVoxel = 0.01f;
        float m_blockWorld = 0.32f;
        float m_detailK = 4.0f;
        bool m_finalized = false;
        TiledAdvancedTSDF m_base;
        TiledAdvancedTSDF m_detail;
        std::unordered_map<BlockKey, uint32_t, BlockKeyHash> m_count;
        std::unordered_map<BlockKey, std::unordered_set<int64_t>, BlockKeyHash> m_occ;
        std::unordered_set<BlockKey, BlockKeyHash> m_dense;
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Build + run the tests to verify they PASS**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8 && ./build/test/vkspatial_tests --gtest_filter='SubmapAdvanced.*'`
Expected: PASS (4 tests). If `BaseReplacedNotAddedInDenseBlocks` ratio bounds are marginal, note the observed ratio in the report (do NOT loosen without cause — a ~5x ratio means dedup is broken).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/SubmapAdvancedTSDF.h test/test_submapAdvancedTsdf.cpp
git commit -m "feat(spatial): SubmapAdvancedTSDF — density-adaptive 2-level detail submaps"
```
(Append trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.)

---

### Task 2: `tsdf_folder_eval --submap`

**Files:**
- Modify: `example2/tsdf_folder_eval.cpp`

**Interfaces:**
- Consumes: `Engine::Spatial::SubmapAdvancedTSDF` (Task 1); the file's existing `frames` (`struct Frame{std::vector<Vector3f> pts,nrm; Vector3f cam;}`), `ctx`, `voxel`, `trunc`, `p2p`, `conf`, `hermite`, `maxPts`, `recon`, arg helpers (`flag`, `floatArg`), and `nextPow2`.
- Produces: a `--submap` reconstruction mode; no new public interface.

- [ ] **Step 1: Add the include**

In `example2/tsdf_folder_eval.cpp`, next to `#include "Engine/Spatial/TiledAdvancedTSDF.h"`:

```cpp
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
```

- [ ] **Step 2: Add the submap branch**

Find where `recon` is produced by the existing single/tiled `if (useTiled) { ... } else { ... }` block (it builds `Engine::Core::Context ctx;` then fills `Engine::Spatial::OrientedPointCloud recon;`). Immediately AFTER `Engine::Core::Context ctx;` is created and BEFORE the `if (useTiled)` branch, insert a submap short-circuit:

```cpp
        const bool submap = flag(argc, argv, "--submap");
        const int blockVoxels = int(floatArg(argc, argv, "--block", 32.0f));
        const float detailK = floatArg(argc, argv, "--detail-k", 4.0f);
        if (submap) {
            Engine::Spatial::SubmapAdvancedTSDF s;
            s.Build(ctx, voxel, trunc, blockVoxels, detailK, 1u << 20, maxPts);
            s.SetIntegrationQuality({3, 4, true});
            s.SetPointToPlane(p2p);
            s.SetConfidenceWeight(conf);
            s.SetHermitePosition(hermite);
            for (const auto &fr: frames) s.AddDensity(fr.pts); // pass 1: density
            s.FinalizeDensity();
            for (const auto &fr: frames) s.Integrate(fr.pts, fr.nrm, fr.cam); // pass 2
            recon = s.ExtractPointCloud(/*merge=*/true);
            std::printf("path      : SUBMAP (base voxel %.4f + detail %.4f); dense blocks %u, "
                        "base tiles %u, detail tiles %u\n",
                        voxel, voxel * 0.5f, s.DenseBlockCount(), s.BaseTileCount(),
                        s.DetailTileCount());
            std::printf("extracted : %zu oriented points\n", recon.points.size());
        } else if (useTiled) {
            // ... existing tiled branch, unchanged ...
```

Wrap the existing `if (useTiled) { ... } else { ... }` as the `else if`/`else` continuation so the single/tiled paths only run when `--submap` is absent. Ensure `recon` is declared once before this whole block (it already is) and the downstream `--out`/RMSE code runs for all three modes.

Note: `recon` must be declared before the `submap`/`useTiled` branching (it already is in the file — `Engine::Spatial::OrientedPointCloud recon;`). Move that declaration above this inserted block if the compiler reports it's declared inside the `else`.

- [ ] **Step 3: Build the tool**

Run: `VULKAN_SDK=/usr/local cmake --build build --target tsdf_folder_eval -j8`
Expected: builds clean.

- [ ] **Step 4: Smoke test — submap on the dragon**

Run:
```bash
SP=/private/tmp/claude-501/-Users-sjy-Desktop-VulkanProject-VkLBVH/2113820b-0832-4916-9850-30862129af68/scratchpad
./build/example2/tsdf_folder_eval --dir scans/dragon --voxel 0.5 --submap 2>&1 | grep -E "path|dense blocks|extracted"
./build/example2/tsdf_folder_eval --dir scans/dragon --voxel 0.5 2>&1 | grep -E "path|extracted"   # non-submap unchanged
```
Expected: `--submap` prints `path : SUBMAP ... dense blocks D, base tiles B, detail tiles T` with `D>0` (dragon has dense regions) and extracted > 0; the non-submap run still prints its SINGLE/TILED path and extracted > 0. Report the dense-block/tile counts and that submap `extracted` ≥ the plain run (detail adds points).

- [ ] **Step 5: Commit**

```bash
git add example2/tsdf_folder_eval.cpp
git commit -m "feat(example2): tsdf_folder_eval --submap (density-adaptive detail submaps)"
```
(Append trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.)

---

## Self-Review

**1. Spec coverage:** 2-level base+detail TiledAdvancedTSDF → Task 1 header. Density blocks + metric (avg pts/occ-voxel ≥ k) → `AddDensity`/`FinalizeDensity`. 2-pass batch → tests + eval Task 2. Precedence-dedup extract → `ExtractPointCloud`. Setters forwarded → `Set*`. Finalize gate → test 4. `--submap`/`--block`/`--detail-k` → Task 2. DenseBlockCount/tile diagnostics → API + eval print. All spec sections covered. ✓

**2. Placeholder scan:** Task 1 fully coded (header + 4 tests). Task 2 shows the exact inserted block + names the existing symbols it reuses; the "existing tiled branch, unchanged" reference is to code already in the file, not omitted new content. No TBD/vague-error placeholders.

**3. Type consistency:** `SubmapAdvancedTSDF` API names match between header (Task 1) and eval use (Task 2): `Build(ctx,baseVoxel,trunc,blockVoxels,detailPtsPerVoxel,tileHashPerTile,maxPointsPerFrame)`, `AddDensity`, `FinalizeDensity`, `Integrate`, `ExtractPointCloud(merge)`, `DenseBlockCount`/`BaseTileCount`/`DetailTileCount`. `OrientedPointCloud.points/.normals`, `TiledAdvancedTSDF` API match the current headers. `detailPtsPerVoxel` (param) vs `detailK`/`--detail-k` (eval local) are consistently mapped.
