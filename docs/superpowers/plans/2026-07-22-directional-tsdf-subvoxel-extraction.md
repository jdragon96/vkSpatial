# DirectionalTSDF Sub-voxel Gradient-refined Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine DirectionalTSDF extracted point positions onto the zero isosurface with a first-order gradient projection in the extract shader (behind a toggle), and prove the quality gain with an extended analytic oracle — without changing output shape, integration, or storage.

**Architecture:** The GPU extract kernel already emits, per voxel, an axis-interpolated crossing position plus a finite-difference gradient. We add an opt-in `g_refine` path that replaces the axis-average position with `p = center − c·voxelSize·grad/dot(grad,grad)` (a first-order Newton step onto the isosurface, unit-correct because the truncation normalization is already baked into `grad`'s magnitude). A `DirectionalTSDF::SetSubvoxelRefine(bool)` flag plumbs the toggle. The existing `example2/tsdf_feature_compare` experiment is extended to run legacy vs refined extraction side by side and score both with per-region position + normal-angle error against analytic shape oracles.

**Tech Stack:** C++17, Eigen, Vulkan compute (GLSL 460), GoogleTest, CMake. Repo: VkLBVH, `Engine::Spatial`.

## Global Constraints

- Output shape unchanged: extraction still returns oriented point clouds (`ExtractedPoint{position, normal, dirMask}` / `OrientedPointCloud`). No integration/host-store/GPU-voxel-layout change.
- Default behavior unchanged: `SetSubvoxelRefine` defaults to `false`; with the default, extraction is bit-identical to today (the `g_refine == 0` legacy branch).
- TSDF value convention: stored voxel value `c = sumDW/sumW` is truncation-normalized (`directional_tsdf_integrate.comp` writes `clamp(sdf/truncation, -1, 1)`), NOT metres. Position projection must use the gradient-magnitude form (no `truncation` constant).
- World unit == 1 mm. Shapes centred at origin. Cube half-extent 1.5, cylinder R=1.5, HZ=1.5, axis +Z (`fixtures::kCubeHalf/kCylRadius/kCylHalfZ`).
- Keep sample counts small (per-view budget ≤ 800, `fixtures::kPerViewBudget`) to avoid the known large-N GPU nondeterminism.
- Every git commit message ends with the trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

- `example2/shape_fixtures.h` (modify) — add `NearestNormal(shape, p)` analytic outward normal + `NormalAngleDeg(a, b)` helper. Render-free (Eigen + std only).
- `test/test_shape_fixtures.cpp` (create) — GTest for the new oracle functions. Links into `vkspatial_tests`; includes `example2/shape_fixtures.h` via the existing `${CMAKE_SOURCE_DIR}` include dir.
- `src/shader/directional_tsdf_extract.comp` (modify) — add `uint g_refine;` push-constant + gradient-projection position branch.
- `src/Engine/Spatial/DirectionalTSDF.h` (modify) — `SetSubvoxelRefine(bool)` setter + `bool m_subvoxelRefine` member.
- `src/Engine/Spatial/DirectionalTSDF.cpp` (modify) — `ExtractPC` gains `refine`; extract dispatch passes the flag.
- `example2/tsdf_feature_compare.cpp` (modify) — run legacy + refined directional extraction, score both (position + normal-angle) per region, print extended table (`--dump`), add a viewer refine toggle.

---

### Task 1: Analytic normal oracle (`NearestNormal`, `NormalAngleDeg`)

**Files:**
- Modify: `example2/shape_fixtures.h` (insert after `NearestDistance`, before the `ClassifyRegion` comment block near line 64-66)
- Test: `test/test_shape_fixtures.cpp` (create)

**Interfaces:**
- Consumes: `fixtures::Shape`, `fixtures::kCubeHalf/kCylRadius/kCylHalfZ/kPi` (existing in `shape_fixtures.h`).
- Produces: `Eigen::Vector3f fixtures::NearestNormal(fixtures::Shape s, const Eigen::Vector3f &p)` — unit outward normal at the nearest surface point. `float fixtures::NormalAngleDeg(const Eigen::Vector3f &a, const Eigen::Vector3f &b)` — angle in degrees.

- [ ] **Step 1: Write the failing test**

Create `test/test_shape_fixtures.cpp`:

```cpp
#include "example2/shape_fixtures.h"

#include <gtest/gtest.h>

using fixtures::Shape;
using Eigen::Vector3f;

namespace {

    TEST(ShapeFixturesNormal, CubePlusXFace) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cube, {fixtures::kCubeHalf, 0.3f, -0.2f});
        EXPECT_NEAR(n.x(), 1.0f, 1e-4f);
        EXPECT_NEAR(n.y(), 0.0f, 1e-4f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CubeMinusZFace) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cube, {0.1f, -0.4f, -fixtures::kCubeHalf});
        EXPECT_NEAR(n.z(), -1.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CubeEdgeBisector) {
        const Vector3f n = fixtures::NearestNormal(
                Shape::Cube, {fixtures::kCubeHalf + 0.05f, fixtures::kCubeHalf + 0.05f, 0.0f});
        EXPECT_NEAR(n.x(), n.y(), 1e-4f);
        EXPECT_GT(n.x(), 0.0f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CylinderSideRadial) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cylinder, {fixtures::kCylRadius, 0.0f, 0.5f});
        EXPECT_NEAR(n.x(), 1.0f, 1e-4f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CylinderCapAxial) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cylinder, {0.3f, -0.2f, fixtures::kCylHalfZ});
        EXPECT_NEAR(n.z(), 1.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, AngleDeg) {
        EXPECT_NEAR(fixtures::NormalAngleDeg(Vector3f::UnitX(), Vector3f::UnitY()), 90.0f, 1e-3f);
        EXPECT_NEAR(fixtures::NormalAngleDeg(Vector3f::UnitX(), Vector3f::UnitX()), 0.0f, 1e-3f);
    }

} // namespace
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake -S . -B build
cmake --build build --target vkspatial_tests -j
```

Expected: FAIL to compile with "`NearestNormal` is not a member of `fixtures`" (function not yet defined).

- [ ] **Step 3: Add the oracle functions**

In `example2/shape_fixtures.h`, immediately after the closing `}` of `NearestDistance` (line 64) and before the `// -------- per-point region classification ...` comment, insert:

```cpp
    // -------- exact outward unit normal at the nearest surface point (oracle for normals) --------
    // At sharp features (cube edges/corners, cylinder rim) the true normal is discontinuous; this
    // returns the box / 2D-box SDF gradient there (the bisector of the adjoining faces). Callers
    // treat Edge-region normal error as a reference number only (see the feature-compare table).

    inline Eigen::Vector3f NearestNormal(Shape s, const Eigen::Vector3f &p) {
        if (s == Shape::Cube) {
            const float qx = std::abs(p.x()) - kCubeHalf;
            const float qy = std::abs(p.y()) - kCubeHalf;
            const float qz = std::abs(p.z()) - kCubeHalf;
            Eigen::Vector3f n((p.x() >= 0.0f ? 1.0f : -1.0f) * std::max(qx, 0.0f),
                              (p.y() >= 0.0f ? 1.0f : -1.0f) * std::max(qy, 0.0f),
                              (p.z() >= 0.0f ? 1.0f : -1.0f) * std::max(qz, 0.0f));
            if (n.norm() > 1e-6f) return n.normalized();
            // Inside (or exactly on a face): nearest face = axis with the largest (closest-to-0) q.
            int axis = 0;
            float best = qx;
            if (qy > best) { best = qy; axis = 1; }
            if (qz > best) { best = qz; axis = 2; }
            Eigen::Vector3f e = Eigen::Vector3f::Zero();
            e[axis] = p[axis] >= 0.0f ? 1.0f : -1.0f;
            return e;
        }
        // Cylinder, axis +Z: 2D box SDF gradient in (radial, axial).
        const float r = std::hypot(p.x(), p.y());
        const float dr = r - kCylRadius;
        const float dz = std::abs(p.z()) - kCylHalfZ;
        const float zsign = p.z() >= 0.0f ? 1.0f : -1.0f;
        Eigen::Vector3f n = Eigen::Vector3f::Zero();
        if (r > 1e-6f) {
            n.x() = (p.x() / r) * std::max(dr, 0.0f);
            n.y() = (p.y() / r) * std::max(dr, 0.0f);
        }
        n.z() = zsign * std::max(dz, 0.0f);
        if (n.norm() > 1e-6f) return n.normalized();
        // Inside: side wall vs cap, whichever gap is closer to zero (the larger negative value).
        if (dr >= dz && r > 1e-6f) return Eigen::Vector3f(p.x() / r, p.y() / r, 0.0f);
        return Eigen::Vector3f(0.0f, 0.0f, zsign);
    }

    // Angle between two vectors in degrees (unit-normalized internally; 0 for a zero input).
    inline float NormalAngleDeg(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
        const float na = a.norm(), nb = b.norm();
        if (na < 1e-12f || nb < 1e-12f) return 0.0f;
        const float c = std::clamp(a.dot(b) / (na * nb), -1.0f, 1.0f);
        return std::acos(c) * 180.0f / kPi;
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cmake --build build --target vkspatial_tests -j
TEST_BIN=$(find build -type f -name vkspatial_tests | head -1)
"$TEST_BIN" --gtest_filter='ShapeFixturesNormal.*'
```

Expected: PASS — `[  PASSED  ] 6 tests.`

- [ ] **Step 5: Commit**

```bash
git add example2/shape_fixtures.h test/test_shape_fixtures.cpp
git commit -m "$(cat <<'EOF'
feat(fixtures): analytic NearestNormal oracle for feature-compare

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extract shader gradient projection + refine toggle plumbing

**Files:**
- Modify: `src/shader/directional_tsdf_extract.comp:19-26` (push constant), `:107-114` (position calc)
- Modify: `src/Engine/Spatial/DirectionalTSDF.h:68` (add setter + member)
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp:34-41` (`ExtractPC`), `:285-287` (dispatch)

**Interfaces:**
- Consumes: existing `IntegrationQuality m_quality`, `ExtractPC` layout, `m_extractKernel->Args(epc)`.
- Produces: `void DirectionalTSDF::SetSubvoxelRefine(bool on)`; shader push-constant field `uint g_refine` (0 = legacy, 1 = refined). No public signature changes to `Integrate`/`PointCloud`.

- [ ] **Step 1: Capture the legacy `--dump` baseline (regression guard)**

Build and record the current numbers (default extraction = legacy):

```bash
cmake --build build --target tsdf_feature_compare -j
APP_BIN=$(find build -type f -name tsdf_feature_compare | head -1)
"$APP_BIN" --shape cube --dump | tee /tmp/subvoxel_baseline_cube.txt
"$APP_BIN" --shape cylinder --dump | tee /tmp/subvoxel_baseline_cyl.txt
```

Expected: two tables print, process exits 0. Keep these files for Step 6.

- [ ] **Step 2: Add the `g_refine` push-constant + refine branch in the shader**

In `src/shader/directional_tsdf_extract.comp`, extend the push-constant block (currently ends with `int g_baseZ;`):

```glsl
layout(push_constant) uniform PC {
    uint  g_numGroups;
    float g_voxelSize;
    uint  g_maxCandidates;
    int   g_baseX;
    int   g_baseY;
    int   g_baseZ;
    uint  g_refine;   // 0 = legacy axis-crossing average, 1 = gradient isosurface projection
};
```

Then replace the single position line `vec3 pos = posSum / float(crossings);` (currently line ~108, just after `Candidate cand;`) with:

```glsl
    vec3 pos;
    if (g_refine == 1u) {
        // First-order projection onto the zero isosurface. dot(grad,grad) >= len^2 > 1e-12 here
        // (len < 1e-6 already returned above). Truncation normalization is baked into |grad|,
        // so no truncation constant is needed. Fall back to the legacy average if the step is
        // implausibly large (weak/degenerate gradient), guarding against divergence.
        vec3 voxelCenter = (vec3(v) + vec3(0.5)) * g_voxelSize;
        vec3 stepv = c * g_voxelSize * grad / dot(grad, grad);
        pos = (length(stepv) > g_voxelSize) ? (posSum / float(crossings)) : (voxelCenter - stepv);
    } else {
        pos = posSum / float(crossings);
    }
```

(Leave the normal `vec3 n = grad / len;` and everything else unchanged.)

- [ ] **Step 3: Add the C++ toggle + push-constant field**

In `src/Engine/Spatial/DirectionalTSDF.h`, right after the `SetIntegrationQuality` method (line 68), add:

```cpp
        // Opt-in sub-voxel extraction: project each extracted point onto the zero isosurface via
        // a first-order gradient step (default off = legacy axis-crossing average). Position only;
        // normals are unchanged. See docs/superpowers/specs/2026-07-22-directional-tsdf-subvoxel-extraction-design.md.
        void SetSubvoxelRefine(bool on) { m_subvoxelRefine = on; }
```

In the same file, next to `IntegrationQuality m_quality;` (line 127), add the member:

```cpp
        bool m_subvoxelRefine = false;
```

In `src/Engine/Spatial/DirectionalTSDF.cpp`, extend `ExtractPC` (lines 34-41) to match the shader:

```cpp
        struct ExtractPC {
            uint32_t numGroups;
            float voxelSize;
            uint32_t maxCandidates;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
            uint32_t refine;
        };
```

And in the extract dispatch (line 285-286), pass the flag:

```cpp
            ExtractPC epc{uint32_t(groupSlots.size()), m_voxelSize, m_maxCandidates,
                          localBase.x(), localBase.y(), localBase.z(),
                          m_subvoxelRefine ? 1u : 0u};
```

- [ ] **Step 4: Build**

```bash
cmake --build build --target vkspatial_tests tsdf_feature_compare -j
```

Expected: builds clean (shaders recompile; the extract `.comp` is picked up by the shader target).

- [ ] **Step 5: Run the existing test suite (no regressions)**

```bash
TEST_BIN=$(find build -type f -name vkspatial_tests | head -1)
"$TEST_BIN" --gtest_filter='ShapeFixturesNormal.*'
```

Expected: PASS (Task 1 tests still green; nothing else touched).

- [ ] **Step 6: Verify legacy `--dump` is unchanged (default refine = false)**

```bash
APP_BIN=$(find build -type f -name tsdf_feature_compare | head -1)
diff <("$APP_BIN" --shape cube --dump) /tmp/subvoxel_baseline_cube.txt && echo "CUBE UNCHANGED"
diff <("$APP_BIN" --shape cylinder --dump) /tmp/subvoxel_baseline_cyl.txt && echo "CYL UNCHANGED"
```

Expected: both `diff`s are empty and print `... UNCHANGED`. This proves the added plumbing does not alter the default (legacy) extraction. If they differ, the `g_refine == 0` branch or the `ExtractPC` layout is wrong — fix before committing.

- [ ] **Step 7: Commit**

```bash
git add src/shader/directional_tsdf_extract.comp src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp
git commit -m "$(cat <<'EOF'
feat(spatial): opt-in sub-voxel gradient-projected DirectionalTSDF extraction

Add a g_refine extract path that projects each point onto the zero
isosurface (p = center - c*voxelSize*grad/dot(grad,grad)); default off
reproduces the legacy axis-crossing average bit-for-bit.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Headless oracle — legacy vs refined position + normal-angle table

**Files:**
- Modify: `example2/tsdf_feature_compare.cpp` — `CompareResult` (lines 93-100), `accumulate`/add `accumulateNormals` (after line 78), `RunCompare` (lines 102-131), `PrintReport` (lines 133-151).

**Interfaces:**
- Consumes: `fixtures::NearestNormal`, `fixtures::NormalAngleDeg` (Task 1); `DirectionalTSDF::SetSubvoxelRefine` (Task 2); existing `RegionStat`, `accumulate`, `regionName`.
- Produces: extended `CompareResult` fields (`dirPtsRefined`, `dirNormals`, `dirNormalsRefined`, `dirStatsRefined`, `dirNormStats`, `dirNormStatsRefined`); `void accumulateNormals(Shape, float, const std::vector<Eigen::Vector3f>&, const std::vector<Eigen::Vector3f>&, std::array<RegionStat, kNumRegions>&)`. `rebuild()` (Task 4) reads these.

- [ ] **Step 1: Extend `CompareResult` and add the normal-error accumulator**

In `example2/tsdf_feature_compare.cpp`, add `accumulateNormals` immediately after `accumulate` (after line 78):

```cpp
    // Per-region normal-angle error (degrees): angle between each extracted normal and the
    // analytic surface normal at that point.
    void accumulateNormals(Shape s, float voxel, const std::vector<Eigen::Vector3f> &pts,
                           const std::vector<Eigen::Vector3f> &normals,
                           std::array<RegionStat, kNumRegions> &stats) {
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const float ang = fixtures::NormalAngleDeg(normals[i], fixtures::NearestNormal(s, pts[i]));
            stats[static_cast<int>(fixtures::ClassifyRegion(s, pts[i], voxel))].add(ang);
        }
    }
```

Extend `CompareResult` (lines 93-100) — keep the existing `dirPts`/`dirStats` (legacy) and add refined + normal fields:

```cpp
    struct CompareResult {
        std::vector<fixtures::View> views;
        std::size_t nInput = 0, maxPerView = 0;
        std::vector<Eigen::Vector3f> simplePts;
        std::vector<Eigen::Vector3f> dirPts;             // legacy directional positions
        std::vector<Eigen::Vector3f> dirNormals;         // legacy directional normals
        std::vector<Eigen::Vector3f> dirPtsRefined;      // refined directional positions
        std::vector<Eigen::Vector3f> dirNormalsRefined;  // refined directional normals
        std::array<RegionStat, kNumRegions> simpleStats{};
        std::array<RegionStat, kNumRegions> dirStats{};            // legacy position error
        std::array<RegionStat, kNumRegions> dirStatsRefined{};     // refined position error
        std::array<RegionStat, kNumRegions> dirNormStats{};        // legacy normal-angle error
        std::array<RegionStat, kNumRegions> dirNormStatsRefined{}; // refined normal-angle error
    };
```

- [ ] **Step 2: Run legacy + refined directional extraction in `RunCompare`**

Replace the DirectionalTSDF block and the accumulate calls in `RunCompare` (lines 118-130) with:

```cpp
        // ---- DirectionalTSDF, legacy extraction (axis-crossing average) ----
        Engine::Spatial::DirectionalTSDF dirLegacy;
        dirLegacy.Build(ctx, voxel, truncation);
        dirLegacy.SetIntegrationQuality({3, 4, true}); // maxDirections=3, dirExponent=4, viewAngleWeight
        dirLegacy.SetSubvoxelRefine(false);
        for (const auto &v : r.views)
            dirLegacy.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirLegacy.PointCloud()) {
            r.dirPts.push_back(e.position);
            r.dirNormals.push_back(e.normal);
        }

        // ---- DirectionalTSDF, refined extraction (sub-voxel gradient projection) ----
        // Same synthetic input is deterministic at this sample budget, so the two integrated
        // volumes match and only the extraction differs. (No public re-extract API exists;
        // a second Build+Integrate is the clean way to get both clouds.)
        Engine::Spatial::DirectionalTSDF dirRefined;
        dirRefined.Build(ctx, voxel, truncation);
        dirRefined.SetIntegrationQuality({3, 4, true});
        dirRefined.SetSubvoxelRefine(true);
        for (const auto &v : r.views)
            dirRefined.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirRefined.PointCloud()) {
            r.dirPtsRefined.push_back(e.position);
            r.dirNormalsRefined.push_back(e.normal);
        }

        accumulate(shape, voxel, r.simplePts, r.simpleStats);
        accumulate(shape, voxel, r.dirPts, r.dirStats);
        accumulate(shape, voxel, r.dirPtsRefined, r.dirStatsRefined);
        accumulateNormals(shape, voxel, r.dirPts, r.dirNormals, r.dirNormStats);
        accumulateNormals(shape, voxel, r.dirPtsRefined, r.dirNormalsRefined, r.dirNormStatsRefined);
        return r;
```

- [ ] **Step 3: Extend `PrintReport` with the two sub-tables**

Replace the body of `PrintReport` (lines 134-151) with:

```cpp
    void PrintReport(const char *shapeName, float voxel, float truncation,
                      const CompareResult &r) {
        std::printf("=== tsdf_feature_compare  shape=%s  voxel=%.3f  truncation=%.3f ===\n",
                    shapeName, voxel, truncation);
        std::printf("views=%zu  nInput=%zu (max/view=%zu)  nSimple=%zu  nDir=%zu  nDirRefined=%zu\n",
                    r.views.size(), r.nInput, r.maxPerView, r.simplePts.size(), r.dirPts.size(),
                    r.dirPtsRefined.size());

        std::printf("[position error mm]\n");
        std::printf("%-7s | %11s %11s | %11s %11s | %11s %11s\n", "region", "Simple.mean",
                    "Simple.max", "Dir.mean", "Dir.max", "DirRef.mean", "DirRef.max");
        std::printf("--------+-------------------------+-------------------------"
                    "+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.simpleStats[reg].count == 0 && r.dirStats[reg].count == 0 &&
                r.dirStatsRefined[reg].count == 0)
                continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f\n", regionName(reg),
                        r.simpleStats[reg].mean(), r.simpleStats[reg].maxErr,
                        r.dirStats[reg].mean(), r.dirStats[reg].maxErr,
                        r.dirStatsRefined[reg].mean(), r.dirStatsRefined[reg].maxErr);
        }

        std::printf("[normal-angle error deg]  (edge = reference only, normal is discontinuous)\n");
        std::printf("%-7s | %11s %11s | %11s %11s\n", "region", "Dir.mean", "Dir.max",
                    "DirRef.mean", "DirRef.max");
        std::printf("--------+-------------------------+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.dirNormStats[reg].count == 0 && r.dirNormStatsRefined[reg].count == 0) continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f\n", regionName(reg),
                        r.dirNormStats[reg].mean(), r.dirNormStats[reg].maxErr,
                        r.dirNormStatsRefined[reg].mean(), r.dirNormStatsRefined[reg].maxErr);
        }
        std::printf("(position mm; angle deg; 1 world unit == 1 mm)\n");
    }
```

- [ ] **Step 4: Build**

```bash
cmake --build build --target tsdf_feature_compare -j
```

Expected: builds clean.

- [ ] **Step 5: Run the extended oracle and check the quality gain**

```bash
APP_BIN=$(find build -type f -name tsdf_feature_compare | head -1)
"$APP_BIN" --shape cube --dump
"$APP_BIN" --shape cylinder --dump
```

Expected (the acceptance criteria):
- Both tables now show a `[position error mm]` block with `DirRef.*` columns and a `[normal-angle error deg]` block.
- **`DirRef.mean` ≤ `Dir.mean` on `flat` and `curved` rows** (sub-voxel projection tightens points onto the surface); **`edge` shows no regression** (`DirRef.mean` ≲ `Dir.mean` within noise).
- Normal-angle columns are informational: if `DirRef` ≈ `Dir` (little change), that is the expected lite outcome and the recorded justification for a future stored-gradient (full) step.

If `DirRef` is worse than `Dir` on flat/curved, the projection sign/units or the fallback clamp is wrong — revisit Task 2 Step 2.

- [ ] **Step 6: Commit**

```bash
git add example2/tsdf_feature_compare.cpp
git commit -m "$(cat <<'EOF'
feat(feature-compare): score legacy vs refined extraction (position + normal)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Windowed viewer — sub-voxel refine toggle

**Files:**
- Modify: `example2/tsdf_feature_compare.cpp` — `CompareState` (lines 161-173), `rebuild` (lines 180-243), the `SetUi` lambda (lines 348-392).

**Interfaces:**
- Consumes: the extended `CompareResult` from Task 3; `PointCloudPass::SetPointSet/SetVisible`.
- Produces: `CompareState.refine` toggling which directional cloud (legacy vs refined) is uploaded to point set 2, plus refined stats shown in the panel.

- [ ] **Step 1: Add `refine` + refined stats to `CompareState`**

In `CompareState` (lines 161-173), add after `bool showInput = false;`:

```cpp
        bool refine = true; // directional set shows the sub-voxel-refined cloud by default
```

and after `std::array<RegionStat, kNumRegions> dirStats{};` add:

```cpp
        std::array<RegionStat, kNumRegions> dirStatsRefined{};
```

- [ ] **Step 2: Upload the selected directional cloud in `rebuild`**

In `rebuild`, after `state.dirStats = r.dirStats;` (line 190) add:

```cpp
        state.dirStatsRefined = r.dirStatsRefined;
```

Replace the "Set 2: DIRECTIONAL extracted" block (lines 216-223) with a refine-aware version:

```cpp
        // Set 2: DIRECTIONAL extracted (legacy or sub-voxel-refined per state.refine).
        const std::vector<Eigen::Vector3f> &dirShow = state.refine ? r.dirPtsRefined : r.dirPts;
        std::vector<PointVertex> dirVerts;
        dirVerts.reserve(dirShow.size());
        for (const auto &p : dirShow) {
            const fixtures::Rgb c = colorFor(p);
            dirVerts.push_back({{p.x(), p.y(), p.z()}, {c.r, c.g, c.b, 255}});
        }
        pass.SetPointSet(2, dirVerts);
```

(`state.nDir` on line 188 still reports the legacy count; leave it — it is only a label.)

- [ ] **Step 3: Add the refine checkbox + refined column to the panel**

In the `SetUi` lambda, after the `show input` checkbox block (lines 371-372), add:

```cpp
            if (ImGui::Checkbox("sub-voxel refine (directional)", &state.refine))
                state.dirty = true; // switches point set 2 -> needs a re-upload
```

Replace the per-region table block (lines 374-385) with one that shows the refined column:

```cpp
            ImGui::SeparatorText("Per-region position error (mm)");
            ImGui::Text("%-7s %11s %11s %11s %11s %11s %11s", "region", "Simple.mean",
                        "Simple.max", "Dir.mean", "Dir.max", "DirRef.mean", "DirRef.max");
            for (int reg = 0; reg < kNumRegions; ++reg) {
                if (state.simpleStats[reg].count == 0 && state.dirStats[reg].count == 0 &&
                    state.dirStatsRefined[reg].count == 0)
                    continue;
                ImGui::Text("%-7s %11.4f %11.4f %11.4f %11.4f %11.4f %11.4f", regionName(reg),
                            state.simpleStats[reg].mean(), state.simpleStats[reg].maxErr,
                            state.dirStats[reg].mean(), state.dirStats[reg].maxErr,
                            state.dirStatsRefined[reg].mean(), state.dirStatsRefined[reg].maxErr);
            }
            ImGui::Text("nInput=%zu  nSimple=%zu  nDir=%zu", state.nInput, state.nSimple,
                        state.nDir);
```

- [ ] **Step 4: Build**

```bash
cmake --build build --target tsdf_feature_compare -j
```

Expected: builds clean.

- [ ] **Step 5: Launch-smoke (headless auto-exit)**

```bash
APP_BIN=$(find build -type f -name tsdf_feature_compare | head -1)
"$APP_BIN" --shape cube --frames 3; echo "exit=$?"
```

Expected: a window opens, renders 3 frames, closes; `exit=0`. No new Vulkan validation errors in the log (the pre-existing `VUID-...-00067`, if present, is out of scope). The `--dump` path is unaffected (Task 3 still passes).

- [ ] **Step 6: Commit**

```bash
git add example2/tsdf_feature_compare.cpp
git commit -m "$(cat <<'EOF'
feat(feature-compare): viewer toggle for sub-voxel-refined directional cloud

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Spec §"단위 근거" (gradient-magnitude projection, no truncation constant) → Task 2 Step 2 (`p = center − c·voxelSize·grad/dot(grad,grad)`, Global Constraints). ✓
- Spec §1 extract shader `g_refine` + fallback clamp → Task 2 Steps 2. ✓
- Spec §2 `SetSubvoxelRefine` + push-constant wiring → Task 2 Step 3. ✓
- Spec §3 `NearestNormal` oracle → Task 1. ✓
- Spec §4 two-cloud (legacy/refined) run, position + normal-angle per region, `--dump` table, viewer toggle → Tasks 3 (headless) + 4 (viewer). ✓
- Spec §"검증": regression (legacy bit-identical) → Task 2 Step 6; numeric acceptance → Task 3 Step 5; launch-smoke → Task 4 Step 5. ✓
- Spec "범위 밖" (stored-gradient full, thin structures, mesh, large-N) → not implemented, by design. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `NearestNormal`/`NormalAngleDeg` signatures identical across Task 1 (def) and Task 3 (use). `SetSubvoxelRefine(bool)` identical across Task 2 (def) and Task 3 (use). `ExtractPC.refine` (C++ `uint32_t`) ↔ `g_refine` (GLSL `uint`) match. `CompareResult` field names (`dirPtsRefined`, `dirStatsRefined`, `dirNormStats`, `dirNormStatsRefined`) used consistently in Tasks 3-4. `accumulateNormals` signature matches its call sites. ✓
