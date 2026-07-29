# Tiled AdvancedTSDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift `AdvancedTSDF`'s single 512³-voxel window limit with lazy spatial tiling, preserving compact-hash + point-to-plane + stored-gradient + A1/A2, so scans grow incrementally (touched tiles only).

**Architecture:** Generalize the proven `TiledCompactDirectionalTSDF` coordinator into a header-only `template<class Backend> TiledDirectionalTSDF`; re-alias the compact version to it and add a thin `TiledAdvancedTSDF` subclass that forwards the A1/A2 setters via a per-tile configuration hook. Wire the folder evaluator to fall back to the tiled path when a scene exceeds one window.

**Tech Stack:** C++17, Eigen, Vulkan compute (existing AdvancedTSDF/CompactDirectionalTSDF shaders, unchanged), GoogleTest, CMake (GLOB-based sources/tests).

## Global Constraints

- **VRAM-only** — no host store, no residency backend, no disk tier (out of scope).
- **No new third-party dependencies** — Eigen + STL only; coordinator is header-only.
- **Behavior-preserving migration** — `TiledCompactDirectionalTSDF`'s public API and results stay identical; the characterization test from Task 1 must stay green through Task 2.
- **No shader changes** — tiling is CPU-side; AdvancedTSDF's `.vert.glsl` shaders are reused as-is.
- **Tile geometry (fixed):** core side `C = kCore = 448` voxels; ghost `G = ceil(trunc/voxel)+1`; enforce `C + 2G ≤ 512`; global voxel origin `O = (0,0,0)`; tile of voxel `v` = `floorDiv(v−O, C)`; tile window `windowMinCorner = (O + tile·C − G)·voxel`.
- **Backend contract:** `Build(ctx, float voxel, float trunc, uint32_t hashCap, uint32_t maxPoints, const Eigen::Vector3f& windowMinCorner)`, `SetIntegrationQuality(const IntegrationQuality&)`, `SetPointToPlane(bool)`, `Integrate(const std::vector<Vector3f>&, const std::vector<Vector3f>&, const Vector3f&)`, `ExtractPointCloud(uint32_t maxCandidates, bool merge) → OrientedPointCloud`, `FilledCount() → uint32_t`. Both `CompactDirectionalTSDF` and `AdvancedTSDF` satisfy this; AdvancedTSDF adds `SetConfidenceWeight(float)` and `SetHermitePosition(bool)`.
- **Git policy:** commit ONLY the files each task creates/modifies; do NOT stage or touch the repository's other pre-existing uncommitted changes. Execute on a feature branch/worktree, not directly on `main`.
- **Build/test commands:** configure with `VULKAN_SDK=/usr/local cmake -S . -B build`; build tests with `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`; run a subset with `./build/test/vkspatial_tests --gtest_filter='<pattern>'`.

---

### Task 1: Characterization test locking TiledCompact behavior

Adds a GoogleTest that pins `TiledCompactDirectionalTSDF`'s current results, so the Task-2 template migration is provably behavior-preserving. (Only an example currently exercises tiling — there is no unit test yet.)

**Files:**
- Create: `test/test_tiledCompactDirectional.cpp`

**Interfaces:**
- Consumes: `Engine::Spatial::TiledCompactDirectionalTSDF` (existing: `Build(ctx, voxel, trunc, hashCapPerTile, maxPtsPerFrame)`, `SetIntegrationQuality`, `SetPointToPlane`, `Integrate`, `ExtractPointCloud(bool merge)`, `FilledCount`, `TileCount`, `CoreVoxels`, `GhostVoxels`, static `floorDiv`), `Engine::Spatial::CompactDirectionalTSDF`, `Engine::Core::Context`, `Engine::Spatial::OrientedPointCloud`.
- Produces: a stable test suite `TiledCompact` that Task 2 keeps green.

- [ ] **Step 1: Write the characterization test**

Create `test/test_tiledCompactDirectional.cpp`:

```cpp
#include "Engine/Core/Context.h"
#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/TiledCompactDirectionalTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using Engine::Spatial::CompactDirectionalTSDF;
using Engine::Spatial::OrientedPointCloud;
using Engine::Spatial::TiledCompactDirectionalTSDF;
using Eigen::Vector3f;

namespace {

    // Axis-aligned +Z plane patch of side `span`, (2*half+1)^2 points, centered at `c`.
    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, const Vector3f &c,
                   float span, int half) {
        pts.clear();
        nrm.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(c.x() + i * step, c.y() + j * step, c.z());
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    // Max over `a` of the nearest-neighbour distance to `b` (brute force; small clouds only).
    float maxNearest(const std::vector<Vector3f> &a, const std::vector<Vector3f> &b) {
        float worst = 0.0f;
        for (const auto &p : a) {
            float best = std::numeric_limits<float>::infinity();
            for (const auto &q : b) best = std::min(best, (p - q).squaredNorm());
            worst = std::max(worst, best);
        }
        return std::sqrt(worst);
    }

} // namespace

// floorDiv rounds toward -inf (documented contract), independent of GPU.
TEST(TiledCompact, FloorDivRoundsTowardNegInf) {
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-1, 448), -1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(0, 448), 0);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(447, 448), 0);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(448, 448), 1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-448, 448), -1);
    EXPECT_EQ(TiledCompactDirectionalTSDF::floorDiv(-449, 448), -2);
}

// A patch that fits inside one tile core AND a single CompactDirectionalTSDF window must extract
// to the same surface through the tiled coordinator as through a bare backend (tiling adds no
// error where it isn't needed). voxel 0.05 -> tile-0 core [0,22.4); center the patch at (6,6,6),
// inside one core on EVERY axis (clear of the z=0 tile boundary + its G=4-voxel ghost band, which
// would otherwise split it across tiles) and inside the bare centered window [-12.8,12.8).
TEST(TiledCompact, TilingMatchesSingleWindowWhereItFits) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 30);

    TiledCompactDirectionalTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const OrientedPointCloud tiledCloud = tiled.ExtractPointCloud(/*merge=*/false);

    CompactDirectionalTSDF single;
    single.Build(ctx, 0.05f, 0.15f, 1u << 20, 1u << 15, Vector3f(-12.8f, -12.8f, -12.8f));
    single.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const OrientedPointCloud singleCloud = single.ExtractPointCloud(1u << 18, /*merge=*/false);

    ASSERT_GT(tiledCloud.points.size(), 100u);
    ASSERT_GT(singleCloud.points.size(), 100u);
    EXPECT_EQ(tiled.TileCount(), 1u);
    // Same lattice -> counts match closely and every point coincides.
    EXPECT_NEAR(double(tiledCloud.points.size()), double(singleCloud.points.size()),
                0.02 * double(singleCloud.points.size()));
    EXPECT_LT(maxNearest(tiledCloud.points, singleCloud.points), 0.05f);
    EXPECT_LT(maxNearest(singleCloud.points, tiledCloud.points), 0.05f);
}

// A plane wider than one core spans >=2 tiles and must extract seam-free (no gap band).
TEST(TiledCompact, PlaneSpanningTilesIsSeamFree) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 200); // x,y in [-15,15]

    TiledCompactDirectionalTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 1.0f));
    const OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/false);

    ASSERT_GT(cloud.points.size(), 1000u);
    EXPECT_GE(tiled.TileCount(), 2u); // crosses the x=0 / y=0 tile boundary
    // Every 0.5-wide x-bin across the interior [-14,14] is populated -> no seam gap.
    const int nbins = 56; // (14 - (-14)) / 0.5
    std::vector<int> bin(nbins, 0);
    for (const auto &p : cloud.points) {
        if (p.x() < -14.0f || p.x() >= 14.0f) continue;
        bin[int((p.x() + 14.0f) / 0.5f)]++;
    }
    for (int b = 0; b < nbins; ++b) EXPECT_GT(bin[b], 0) << "empty x-bin " << b << " (seam gap)";
}
```

- [ ] **Step 2: Configure + build the test target**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`
Expected: builds clean (new file is GLOB-picked).

- [ ] **Step 3: Run the new tests to verify they PASS on the current implementation**

Run: `./build/test/vkspatial_tests --gtest_filter='TiledCompact.*'`
Expected: PASS (3 tests). This is a characterization test: it passes now and must keep passing after the Task-2 migration.

- [ ] **Step 4: Commit**

```bash
git add test/test_tiledCompactDirectional.cpp
git commit -m "test: characterization tests locking TiledCompactDirectionalTSDF behavior"
```

---

### Task 2: Generalize into `TiledDirectionalTSDF<Backend>`; migrate the compact alias

Moves the tiling logic into a header-only template, re-aliases `TiledCompactDirectionalTSDF` to it, and deletes the now-empty `.cpp`. Task-1's test is the guard.

**Files:**
- Create: `src/Engine/Spatial/TiledDirectionalTSDF.h`
- Modify (replace contents): `src/Engine/Spatial/TiledCompactDirectionalTSDF.h`
- Delete: `src/Engine/Spatial/TiledCompactDirectionalTSDF.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Context`, `Engine::Spatial::IntegrationQuality` (from `DirectionalIntegrationQuality.h`), `Engine::Spatial::OrientedPointCloud`, and any type satisfying the Backend contract (Global Constraints).
- Produces: `template<class Backend> class Engine::Spatial::TiledDirectionalTSDF` with public `Build(ctx, voxel, trunc, hashCapPerTile=1<<22, maxPtsPerFrame=1<<17)`, `SetIntegrationQuality`, `SetPointToPlane`, `Integrate`, `ExtractPointCloud(bool merge=true)`, `FilledCount`, `TileCount`, `CoreVoxels`, `GhostVoxels`, static `floorDiv`; `protected std::function<void(Backend&)> m_configureHook`. Alias `using TiledCompactDirectionalTSDF = TiledDirectionalTSDF<CompactDirectionalTSDF>;`.

- [ ] **Step 1: Create the templated coordinator header**

Create `src/Engine/Spatial/TiledDirectionalTSDF.h`:

```cpp
#pragma once

#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include "Engine/Spatial/OrientedPointCloud.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {

    // CPU coordinator that lifts a single-window directional-TSDF Backend's 512^3 limit by TILING
    // space: it partitions the world voxel grid into fixed cubic tiles, each backed by one Backend
    // instance (a 512^3 window). Only touched tiles are ever allocated (lazy, on first
    // integration), so memory is proportional to the scanned surface, and the scene can grow
    // indefinitely within VRAM.
    //
    // GEOMETRY (fixed): core side C = kCore = 448 voxels; per-tile 512^3 window = core + ghost
    // margin G on every side (G = ceil(trunc/voxel)+1, covers the full truncation band); requires
    // C + 2G <= 512. Global voxel origin O = 0; tile of voxel v is floorDiv(v - O, C) per axis.
    //
    // GHOST ROUTING: a point near a tile boundary is integrated into BOTH the owning tile and the
    // adjacent tile(s), so every tile's core carries the full truncation band (no seam).
    // CORE-ONLY EXTRACTION: each tile emits only points whose voxel lies in its own core, so the
    // ghost overlap never produces duplicates across tiles.
    //
    // Backend contract: Build(ctx, voxel, trunc, hashCap, maxPoints, windowMinCorner),
    // SetIntegrationQuality, SetPointToPlane, Integrate(points, normals, cameraPos),
    // ExtractPointCloud(maxCandidates, merge) -> OrientedPointCloud, FilledCount. Backend-specific
    // configuration (e.g. AdvancedTSDF's A1/A2 setters) is forwarded via m_configureHook.
    template <class Backend>
    class TiledDirectionalTSDF {
    public:
        TiledDirectionalTSDF() = default;
        virtual ~TiledDirectionalTSDF() = default;

        // Store params + ctx. Tiles are created lazily on first integration. hashCapacityPerTile is
        // the per-tile Backend hash size; maxPointsPerFrame the per-tile per-Integrate point cap.
        void Build(Engine::Core::Context &ctx,
                   float voxelSize,
                   float truncation,
                   uint32_t hashCapacityPerTile = 1u << 22,
                   uint32_t maxPointsPerFrame = 1u << 17) {
            m_ctx = &ctx;
            m_voxelSize = voxelSize;
            m_truncation = truncation;
            m_hashCapacityPerTile = hashCapacityPerTile;
            m_maxPointsPerFrame = maxPointsPerFrame;
            m_origin = Eigen::Vector3i::Zero();
            // G must cover the full truncation band (band radius = ceil(trunc/voxel) voxels), +1.
            m_ghost = static_cast<int>(std::ceil(truncation / voxelSize)) + 1;
            if (kCore + 2 * m_ghost > 512) {
                throw std::runtime_error(
                        "TiledDirectionalTSDF: C + 2G exceeds the 512^3 window "
                        "(truncation too large for voxelSize)");
            }
            m_tiles.clear();
        }

        // Applied to each tile's Backend on creation.
        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Integrate SDF form, forwarded to every tile (default true = point-to-plane).
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // Route each point to its owning tile plus any adjacent tile whose ghost band it falls in,
        // then integrate the per-tile sublists (lazily creating+building touched tiles).
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero()) {
            const size_t n = std::min(points.size(), normals.size());
            if (n == 0) return;

            struct SubList {
                std::vector<Eigen::Vector3f> pts, nrm;
            };
            std::unordered_map<TileKey, SubList, TileKeyHash> routed;

            for (size_t i = 0; i < n; ++i) {
                const Eigen::Vector3f &p = points[i];
                const Eigen::Vector3i v(static_cast<int>(std::floor(p.x() / m_voxelSize)),
                                        static_cast<int>(std::floor(p.y() / m_voxelSize)),
                                        static_cast<int>(std::floor(p.z() / m_voxelSize)));
                const Eigen::Vector3i home = tileOf(v);
                const Eigen::Vector3i local = (v - m_origin) - home * kCore; // 0..C-1 per axis

                int offs[3][2];
                int nOff[3];
                for (int a = 0; a < 3; ++a) {
                    offs[a][0] = 0;
                    nOff[a] = 1;
                    if (local[a] < m_ghost) {
                        offs[a][1] = -1;
                        nOff[a] = 2;
                    } else if (local[a] >= kCore - m_ghost) {
                        offs[a][1] = +1;
                        nOff[a] = 2;
                    }
                }

                for (int ix = 0; ix < nOff[0]; ++ix)
                    for (int iy = 0; iy < nOff[1]; ++iy)
                        for (int iz = 0; iz < nOff[2]; ++iz) {
                            const TileKey key{home.x() + offs[0][ix], home.y() + offs[1][iy],
                                              home.z() + offs[2][iz]};
                            SubList &s = routed[key];
                            s.pts.push_back(p);
                            s.nrm.push_back(normals[i]);
                        }
            }

            for (auto &kv : routed) {
                Backend *tile = tileFor(kv.first);
                tile->Integrate(kv.second.pts, kv.second.nrm, cameraPos);
            }
        }

        // Extract each tile, KEEP ONLY points whose voxel lies in that tile's core (drop ghost
        // duplicates), and concatenate. merge is forwarded to Backend::ExtractPointCloud.
        OrientedPointCloud ExtractPointCloud(bool merge = true) const {
            OrientedPointCloud out;
            for (const auto &kv : m_tiles) {
                const TileKey &key = kv.first;
                const Eigen::Vector3i tile(key.x, key.y, key.z);
                const Eigen::Vector3i coreMin = m_origin + tile * kCore;
                const Eigen::Vector3i coreMax = coreMin + Eigen::Vector3i::Constant(kCore);

                const OrientedPointCloud tileCloud = kv.second->ExtractPointCloud(1u << 21, merge);
                const size_t m = std::min(tileCloud.points.size(), tileCloud.normals.size());
                for (size_t i = 0; i < m; ++i) {
                    const Eigen::Vector3f &p = tileCloud.points[i];
                    const Eigen::Vector3i v(static_cast<int>(std::floor(p.x() / m_voxelSize)),
                                            static_cast<int>(std::floor(p.y() / m_voxelSize)),
                                            static_cast<int>(std::floor(p.z() / m_voxelSize)));
                    if (v.x() >= coreMin.x() && v.x() < coreMax.x() && v.y() >= coreMin.y() &&
                        v.y() < coreMax.y() && v.z() >= coreMin.z() && v.z() < coreMax.z()) {
                        out.points.push_back(p);
                        out.normals.push_back(tileCloud.normals[i]);
                    }
                }
            }
            return out;
        }

        // Sum of tiles' FilledCount. NOTE: ghost overlap makes this slightly MORE than the true
        // occupied-voxel set -- that surplus is the honest tiling overhead.
        uint32_t FilledCount() const {
            uint32_t total = 0;
            for (const auto &kv : m_tiles) total += kv.second->FilledCount();
            return total;
        }

        uint32_t TileCount() const { return static_cast<uint32_t>(m_tiles.size()); }

        // Fixed core side C and the derived ghost margin G (0 before Build). Diagnostics.
        int CoreVoxels() const { return kCore; }
        int GhostVoxels() const { return m_ghost; }

        // floor(a / b) with rounding toward -inf for negatives (callers pass b = C > 0).
        static int floorDiv(int a, int b) {
            int q = a / b;
            int r = a % b;
            if (r != 0 && ((r < 0) != (b < 0))) --q;
            return q;
        }

    protected:
        // Extension point: applied to each tile right after the common config (quality + p2p), on
        // creation. Subclasses set this to forward Backend-specific settings (e.g. A1/A2).
        std::function<void(Backend &)> m_configureHook;

    private:
        static constexpr int kCore = 448; // C: voxels per tile core axis

        struct TileKey {
            int x, y, z;
            bool operator==(const TileKey &o) const { return x == o.x && y == o.y && z == o.z; }
        };
        struct TileKeyHash {
            std::size_t operator()(const TileKey &k) const {
                std::size_t h = std::hash<int>()(k.x);
                h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
                return h;
            }
        };

        Eigen::Vector3i tileOf(const Eigen::Vector3i &v) const {
            return Eigen::Vector3i(floorDiv(v.x() - m_origin.x(), kCore),
                                   floorDiv(v.y() - m_origin.y(), kCore),
                                   floorDiv(v.z() - m_origin.z(), kCore));
        }

        Backend *tileFor(const TileKey &key) {
            auto it = m_tiles.find(key);
            if (it != m_tiles.end()) return it->second.get();

            const Eigen::Vector3i tile(key.x, key.y, key.z);
            const Eigen::Vector3i originVoxel =
                    m_origin + tile * kCore - Eigen::Vector3i::Constant(m_ghost);
            const Eigen::Vector3f windowMinCorner = originVoxel.cast<float>() * m_voxelSize;

            auto tsdf = std::make_unique<Backend>();
            tsdf->Build(*m_ctx, m_voxelSize, m_truncation, m_hashCapacityPerTile,
                        m_maxPointsPerFrame, windowMinCorner);
            tsdf->SetIntegrationQuality(m_quality);
            tsdf->SetPointToPlane(m_pointToPlane);
            if (m_configureHook) m_configureHook(*tsdf);
            Backend *ptr = tsdf.get();
            m_tiles.emplace(key, std::move(tsdf));
            return ptr;
        }

        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.001f;
        float m_truncation = 0.003f;
        uint32_t m_hashCapacityPerTile = 1u << 22;
        uint32_t m_maxPointsPerFrame = 1u << 17;
        int m_ghost = 0;                                    // G (set in Build)
        Eigen::Vector3i m_origin = Eigen::Vector3i::Zero(); // O
        IntegrationQuality m_quality;
        bool m_pointToPlane = true;

        std::unordered_map<TileKey, std::unique_ptr<Backend>, TileKeyHash> m_tiles;
    };

} // namespace Engine::Spatial
```

- [ ] **Step 2: Replace `TiledCompactDirectionalTSDF.h` with the alias**

Replace the entire contents of `src/Engine/Spatial/TiledCompactDirectionalTSDF.h` with:

```cpp
#pragma once

#include "Engine/Spatial/CompactDirectionalTSDF.h"
#include "Engine/Spatial/TiledDirectionalTSDF.h"

namespace Engine::Spatial {

    // TiledCompactDirectionalTSDF is the compact-backend instantiation of the generalized tiling
    // coordinator. See TiledDirectionalTSDF.h for geometry, ghost routing, and extraction.
    using TiledCompactDirectionalTSDF = TiledDirectionalTSDF<CompactDirectionalTSDF>;

} // namespace Engine::Spatial
```

- [ ] **Step 3: Delete the migrated source file**

```bash
git rm src/Engine/Spatial/TiledCompactDirectionalTSDF.cpp
```

- [ ] **Step 4: Reconfigure (drops the deleted .cpp from the GLOB) and build**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`
Expected: builds clean; no reference to the deleted `.cpp` remains.

- [ ] **Step 5: Run the characterization + compact-backend tests to verify PASS (migration is behavior-preserving)**

Run: `./build/test/vkspatial_tests --gtest_filter='TiledCompact.*:CompactDirectional.*'`
Expected: PASS — identical results to Task 1 (the migration changed structure, not behavior).

- [ ] **Step 6: Commit**

```bash
git add src/Engine/Spatial/TiledDirectionalTSDF.h src/Engine/Spatial/TiledCompactDirectionalTSDF.h
git rm src/Engine/Spatial/TiledCompactDirectionalTSDF.cpp
git commit -m "refactor(spatial): generalize tiling into TiledDirectionalTSDF<Backend>; alias TiledCompact"
```

---

### Task 3: `TiledAdvancedTSDF` subclass + tests

Adds the AdvancedTSDF-backed tiled class (with A1/A2 forwarding through the hook) and its correctness tests.

**Files:**
- Create: `src/Engine/Spatial/TiledAdvancedTSDF.h`
- Create: `test/test_tiledAdvancedTsdf.cpp`

**Interfaces:**
- Consumes: `Engine::Spatial::TiledDirectionalTSDF<Backend>` (Task 2), `Engine::Spatial::AdvancedTSDF` (existing: adds `SetConfidenceWeight(float)`, `SetHermitePosition(bool)`).
- Produces: `class Engine::Spatial::TiledAdvancedTSDF : public TiledDirectionalTSDF<AdvancedTSDF>` with extra `SetConfidenceWeight(float)` / `SetHermitePosition(bool)` (defaults λ=0.5, hermite=false — matching AdvancedTSDF). Must be configured before `Integrate` (hook applies at tile creation).

- [ ] **Step 1: Write the failing tests**

Create `test/test_tiledAdvancedTsdf.cpp`:

```cpp
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using Engine::Spatial::AdvancedTSDF;
using Engine::Spatial::OrientedPointCloud;
using Engine::Spatial::TiledAdvancedTSDF;
using Eigen::Vector3f;

namespace {

    void makePlane(std::vector<Vector3f> &pts, std::vector<Vector3f> &nrm, const Vector3f &c,
                   float span, int half) {
        pts.clear();
        nrm.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                pts.emplace_back(c.x() + i * step, c.y() + j * step, c.z());
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

    float maxNearest(const std::vector<Vector3f> &a, const std::vector<Vector3f> &b) {
        float worst = 0.0f;
        for (const auto &p : a) {
            float best = std::numeric_limits<float>::infinity();
            for (const auto &q : b) best = std::min(best, (p - q).squaredNorm());
            worst = std::max(worst, best);
        }
        return std::sqrt(worst);
    }

} // namespace

// Tiling adds no error where a single AdvancedTSDF window already fits. Patch centered at (6,6,6):
// inside tile-0's core on every axis (clear of tile boundaries + ghost bands, so TileCount==1) and
// inside the centered default window [-12.8,12.8) at voxel 0.05.
TEST(TiledAdvanced, TilingMatchesSingleWindowWhereItFits) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 30);

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const OrientedPointCloud tiledCloud = tiled.ExtractPointCloud(/*merge=*/false);

    AdvancedTSDF single;
    single.Build(ctx, 0.05f, 0.15f); // default centered window covers [-12.8, 12.8)
    single.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const OrientedPointCloud singleCloud = single.ExtractPointCloud(1u << 18, /*merge=*/false);

    ASSERT_GT(tiledCloud.points.size(), 100u);
    ASSERT_GT(singleCloud.points.size(), 100u);
    EXPECT_EQ(tiled.TileCount(), 1u);
    EXPECT_NEAR(double(tiledCloud.points.size()), double(singleCloud.points.size()),
                0.02 * double(singleCloud.points.size()));
    EXPECT_LT(maxNearest(tiledCloud.points, singleCloud.points), 0.05f);
    EXPECT_LT(maxNearest(singleCloud.points, tiledCloud.points), 0.05f);
}

// A plane wider than one core spans >=2 tiles with no seam gap.
TEST(TiledAdvanced, PlaneSpanningTilesIsSeamFree) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(0.0f, 0.0f, 0.0f), 30.0f, 200); // x,y in [-15,15]

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(pts, nrm, Vector3f(0.0f, 0.0f, 1.0f));
    const OrientedPointCloud cloud = tiled.ExtractPointCloud(/*merge=*/false);

    ASSERT_GT(cloud.points.size(), 1000u);
    EXPECT_GE(tiled.TileCount(), 2u);
    const int nbins = 56;
    std::vector<int> bin(nbins, 0);
    for (const auto &p : cloud.points) {
        if (p.x() < -14.0f || p.x() >= 14.0f) continue;
        bin[int((p.x() + 14.0f) / 0.5f)]++;
    }
    for (int b = 0; b < nbins; ++b) EXPECT_GT(bin[b], 0) << "empty x-bin " << b << " (seam gap)";
}

// Only tiles a point (or its ghost) touches are allocated; two separated patches -> exactly 2.
TEST(TiledAdvanced, OnlyTouchedTilesAllocated) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> a, an, b, bn;
    makePlane(a, an, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 20);    // tile (0,0,0)
    makePlane(b, bn, Vector3f(106.0f, 6.0f, 6.0f), 4.0f, 20);  // x~2120 vox -> tile (4,0,0)

    TiledAdvancedTSDF tiled;
    tiled.Build(ctx, 0.05f, 0.15f);
    tiled.Integrate(a, an, Vector3f(6.0f, 6.0f, 7.0f));
    tiled.Integrate(b, bn, Vector3f(106.0f, 6.0f, 7.0f));
    EXPECT_EQ(tiled.TileCount(), 2u);
}

// A1 (confidence weight) reaches the tiles: changing it changes the extracted surface; A2
// (Hermite) runs and still produces a surface.
TEST(TiledAdvanced, A1A2SettersReachTiles) {
    Engine::Core::Context ctx;
    std::vector<Vector3f> pts, nrm;
    makePlane(pts, nrm, Vector3f(6.0f, 6.0f, 6.0f), 4.0f, 40);

    TiledAdvancedTSDF off;
    off.Build(ctx, 0.05f, 0.15f);
    off.SetConfidenceWeight(0.0f);
    off.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const size_t nOff = off.ExtractPointCloud(/*merge=*/false).points.size();

    TiledAdvancedTSDF on;
    on.Build(ctx, 0.05f, 0.15f);
    on.SetConfidenceWeight(0.5f);
    on.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    const size_t nOn = on.ExtractPointCloud(/*merge=*/false).points.size();

    ASSERT_GT(nOff, 0u);
    ASSERT_GT(nOn, 0u);
    EXPECT_NE(nOff, nOn) << "SetConfidenceWeight had no effect -> A1 not reaching tiles";

    TiledAdvancedTSDF herm;
    herm.Build(ctx, 0.05f, 0.15f);
    herm.SetHermitePosition(true);
    herm.Integrate(pts, nrm, Vector3f(6.0f, 6.0f, 7.0f));
    EXPECT_GT(herm.ExtractPointCloud(/*merge=*/false).points.size(), 0u);
}
```

- [ ] **Step 2: Run the tests to verify they FAIL (class does not exist yet)**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`
Expected: BUILD FAILS with `fatal error: 'Engine/Spatial/TiledAdvancedTSDF.h' file not found`.

- [ ] **Step 3: Create the `TiledAdvancedTSDF` subclass**

Create `src/Engine/Spatial/TiledAdvancedTSDF.h`:

```cpp
#pragma once

#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/TiledDirectionalTSDF.h"

namespace Engine::Spatial {

    // AdvancedTSDF-backed tiled TSDF: same geometry/routing/extraction as TiledDirectionalTSDF, but
    // each tile is an AdvancedTSDF (compact hash + point-to-plane + stored-gradient). Adds the A1
    // (confidence weight) and A2 (Hermite position) setters, forwarded to every tile via the
    // configure hook. Configure these BEFORE Integrate -- the hook is applied when a tile is first
    // created (during Integrate).
    class TiledAdvancedTSDF : public TiledDirectionalTSDF<AdvancedTSDF> {
    public:
        TiledAdvancedTSDF() {
            m_configureHook = [this](AdvancedTSDF &tile) {
                tile.SetConfidenceWeight(m_confWeight);
                tile.SetHermitePosition(m_hermite);
            };
        }

        // A1: surface-proximity confidence weight lambda in [0,1]; 0 = off. Matches AdvancedTSDF.
        void SetConfidenceWeight(float lambda) { m_confWeight = lambda; }

        // A2: cubic-Hermite zero-crossing position instead of linear. false = linear (default).
        void SetHermitePosition(bool on) { m_hermite = on; }

    private:
        float m_confWeight = 0.5f; // A1 default matches AdvancedTSDF
        bool m_hermite = false;    // A2 default matches AdvancedTSDF
    };

} // namespace Engine::Spatial
```

- [ ] **Step 4: Build and run the tests to verify they PASS**

Run: `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8 && ./build/test/vkspatial_tests --gtest_filter='TiledAdvanced.*'`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Spatial/TiledAdvancedTSDF.h test/test_tiledAdvancedTsdf.cpp
git commit -m "feat(spatial): TiledAdvancedTSDF (tiled AdvancedTSDF with A1/A2 forwarding) + tests"
```

---

### Task 4: eval tool — auto tiled fallback for oversized scenes

Replaces `tsdf_folder_eval`'s hard "voxel too fine" error with an automatic `TiledAdvancedTSDF` path when the scene exceeds one window; adds `--single` to force the single-window path.

**Files:**
- Modify: `example2/tsdf_folder_eval.cpp` (the `axisVox > 512` block and the integrate/extract section)

**Interfaces:**
- Consumes: `Engine::Spatial::TiledAdvancedTSDF` (Task 3), plus the file's existing `voxel`, `trunc`, `p2p`, `conf`, `hermite`, `hashCap`, `maxPts`, `axisVox`, `frames`, `recon` locals.
- Produces: no new public interface; the tool now reconstructs oversized scenes.

- [ ] **Step 1: Add the include**

In `example2/tsdf_folder_eval.cpp`, add next to the existing `#include "Engine/Spatial/AdvancedTSDF.h"`:

```cpp
#include "Engine/Spatial/TiledAdvancedTSDF.h"
```

- [ ] **Step 2: Replace the hard error + single-window build with a branch**

Find the block that currently reads (the `axisVox > 512` guard added earlier):

```cpp
        if (axisVox > 512) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3.\n"
                         "  use --voxel >= %.4f  (object max-span %.3f / %d), or tile the scene "
                         "(TiledCompactDirectionalTSDF).\n",
                         voxel, axisVox, minVoxel, maxSpan, 512 - 2 * margin);
            return 3;
        }
```

Replace it with a `--single` guard (keep the error only when forced):

```cpp
        const bool forceSingle = flag(argc, argv, "--single");
        if (axisVox > 512 && forceSingle) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3 (--single).\n"
                         "  drop --single to auto-tile, or use --voxel >= %.4f.\n",
                         voxel, axisVox, minVoxel);
            return 3;
        }
        const bool useTiled = axisVox > 512; // exceeds one window -> tile
```

Then find the integrate/extract section (which currently unconditionally builds a single `AdvancedTSDF`):

```cpp
        // Integrate + extract.
        Engine::Core::Context ctx;
        Engine::Spatial::AdvancedTSDF tsdf;
        tsdf.Build(ctx, voxel, trunc, hashCap, maxPts, windowMinCorner);
        tsdf.SetIntegrationQuality({3, 4, true});
        tsdf.SetPointToPlane(p2p);
        tsdf.SetConfidenceWeight(conf);
        tsdf.SetHermitePosition(hermite);
        for (const auto &fr: frames) tsdf.Integrate(fr.pts, fr.nrm, fr.cam);
        const auto recon = tsdf.ExtractPointCloud(1u << 21, /*merge=*/true);
        std::printf("integrated: %zu frames → %u occupied entries\n", frames.size(),
                    tsdf.FilledCount());
        std::printf("extracted : %zu oriented points\n", recon.points.size());
```

Replace it with a branch producing `recon`:

```cpp
        // Integrate + extract (single window if it fits, else tiled).
        Engine::Core::Context ctx;
        OrientedPointCloud recon;
        if (useTiled) {
            std::printf("path      : TILED (scene exceeds one 512^3 window)\n");
            Engine::Spatial::TiledAdvancedTSDF tiled;
            tiled.Build(ctx, voxel, trunc, /*hashCapPerTile=*/1u << 21, /*maxPtsPerFrame=*/maxPts);
            tiled.SetIntegrationQuality({3, 4, true});
            tiled.SetPointToPlane(p2p);
            tiled.SetConfidenceWeight(conf);
            tiled.SetHermitePosition(hermite);
            for (const auto &fr: frames) tiled.Integrate(fr.pts, fr.nrm, fr.cam);
            recon = tiled.ExtractPointCloud(/*merge=*/true);
            std::printf("integrated: %zu frames → %u tiles, %u occupied entries (ghost-inflated)\n",
                        frames.size(), tiled.TileCount(), tiled.FilledCount());
        } else {
            std::printf("path      : SINGLE 512^3 window\n");
            Engine::Spatial::AdvancedTSDF tsdf;
            tsdf.Build(ctx, voxel, trunc, hashCap, maxPts, windowMinCorner);
            tsdf.SetIntegrationQuality({3, 4, true});
            tsdf.SetPointToPlane(p2p);
            tsdf.SetConfidenceWeight(conf);
            tsdf.SetHermitePosition(hermite);
            for (const auto &fr: frames) tsdf.Integrate(fr.pts, fr.nrm, fr.cam);
            recon = tsdf.ExtractPointCloud(1u << 21, /*merge=*/true);
            std::printf("integrated: %zu frames → %u occupied entries\n", frames.size(),
                        tsdf.FilledCount());
        }
        std::printf("extracted : %zu oriented points\n", recon.points.size());
```

Note: add `#include "Engine/Spatial/OrientedPointCloud.h"` at the top if not already transitively available (it is, via AdvancedTSDF.h → OrientedPointCloud.h; add explicitly for clarity).

- [ ] **Step 3: Build the tool**

Run: `VULKAN_SDK=/usr/local cmake --build build --target tsdf_folder_eval -j8`
Expected: builds clean.

- [ ] **Step 4: Smoke test — oversized scene now reconstructs; --single still errors; small scene unchanged**

Run (SP = the session scratchpad holding the earlier `scan_dragon` / `scan_cube` folders):

```bash
SP=/private/tmp/claude-501/-Users-sjy-Desktop-VulkanProject-VkLBVH/2113820b-0832-4916-9850-30862129af68/scratchpad
./build/example2/tsdf_folder_eval --dir "$SP/scan_dragon" --voxel 0.5 2>&1 | grep -E "path|tiles|extracted"
./build/example2/tsdf_folder_eval --dir "$SP/scan_dragon" --voxel 0.01 2>&1 | grep -E "path|tiles|extracted"
./build/example2/tsdf_folder_eval --dir "$SP/scan_dragon" --voxel 0.01 --single 2>&1 | grep -E "error|too fine"; echo "single exit=$?"
./build/example2/tsdf_folder_eval --dir "$SP/scan_cube" 2>&1 | grep -E "path|extracted"
```

Expected:
- `--voxel 0.5` (Dragon, ~393 vox/axis ≤ 512): `path : SINGLE 512^3 window`, extracted > 0.
- `--voxel 0.01` (Dragon, ~19125 vox/axis): `path : TILED …`, `… N tiles …`, extracted > 0 (previously produced 0 / errored).
- `--voxel 0.01 --single`: prints the "too fine" error, non-zero exit.
- `scan_cube` (auto): `path : SINGLE 512^3 window`, extracted > 0 (unchanged).

- [ ] **Step 5: Commit**

```bash
git add example2/tsdf_folder_eval.cpp
git commit -m "feat(example2): tsdf_folder_eval auto-tiles oversized scenes (TiledAdvancedTSDF); --single opt-out"
```

---

## Self-Review

**1. Spec coverage:**
- Templated coordinator `TiledDirectionalTSDF<Backend>` → Task 2. ✓
- Migrate `TiledCompactDirectionalTSDF` to alias, delete `.cpp` → Task 2. ✓
- `TiledAdvancedTSDF` subclass + A1/A2 via configure hook → Task 3. ✓
- Tile geometry (C=448, G, C+2G≤512, O=0, floorDiv) → Global Constraints + Task 2 code. ✓
- Ghost routing + core-only extraction → Task 2 `Integrate`/`ExtractPointCloud`. ✓
- Tests: tiling==single-where-fits, seam-free, lazy allocation, A1/A2 reach tiles → Task 3 (+ Task 1 for the compact backend). ✓
- Backend contract satisfied by both → Global Constraints; verified identical `Build`/`ExtractPointCloud` signatures. ✓
- eval auto-tiled fallback + `--single` → Task 4. ✓
- Behavior-preserving migration guard → Task 1 characterization test kept green in Task 2. ✓
- Out-of-scope tiers (residency/disk/multi-res) → not built. ✓

Gap note: the spec claimed "re-applied to existing tiles" for settings; the plan matches the existing codebase behavior (settings applied at tile creation — configure before Integrate) to keep the migration behavior-preserving. This is a deliberate simplification (YAGNI); tests set A1/A2 before Integrate accordingly.

**2. Placeholder scan:** No TBD/TODO/"handle errors"/"similar to". Every code step carries complete code; every run step has exact command + expected output.

**3. Type consistency:** `TiledDirectionalTSDF<Backend>` public API (Task 2) matches the alias/subclass use in Tasks 3–4 and the tests. `ExtractPointCloud(bool merge)` on the tiled coordinator vs `ExtractPointCloud(uint32_t, bool)` on the backends is intentional and used correctly in each context. `SetConfidenceWeight(float)`/`SetHermitePosition(bool)` names match AdvancedTSDF. `hashCap`/`maxPts`/`windowMinCorner`/`axisVox`/`recon` in Task 4 match the identifiers already in `tsdf_folder_eval.cpp`.
