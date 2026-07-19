# Directional TSDF Quality Core (multi-direction + confidence + merge/split) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `Engine::Spatial::DirectionalTSDF` reconstruction quality (spec §7) via multi-direction soft integration + view-angle confidence weighting + merge/split refinement — as **opt-in** settings whose defaults reproduce today's single-direction behavior bit-for-bit, with the quality win proven by an interproximal benchmark.

**Architecture:** A new `IntegrationQuality { maxDirections, dirExponent, viewAngleWeight }` setting (default `{1, 4, false}` = current behavior) flows into the CPU IntegrationWriteSet builder and the integrate compute shader. A shared top-K direction selector (`|n_axis|^p`, relative-weight ≥ 0.05, ≤ K dirs) is computed identically on CPU (for residency) and GPU (for accumulation). Extraction is unchanged; the CPU candidate merge gains an explicit 60° strong-split.

**Tech Stack:** C++17, Vulkan + VMA, Eigen, GLSL compute, GoogleTest. macOS/MoltenVK.

## Global Constraints

- **Opt-in, backward-compatible by default.** `IntegrationQuality` defaults `maxDirections=1, dirExponent=4, viewAngleWeight=false`. With these defaults the integrate path MUST be bit-for-bit identical to today's (`sumW += SCALE`, single `dominantAxis` layer). Every existing `test/test_directionalTSDF.cpp` / `test/test_directionalTSDFEval.cpp` test MUST pass UNCHANGED — do not edit them.
- **CPU/GPU direction selection must match exactly.** The top-K selector and the relative weights `r_d/r_max` must be computed with identical arithmetic on CPU (`DirectionalTSDF.cpp`) and GPU (`directional_tsdf_integrate.comp`) — same candidate order, same integer-power (`p=4` via repeated multiply, NOT `pow()`), same `minRelWeight=0.05f` cutoff, same tie-break (x, then y, then z), mirroring the existing `dominantAxis`/`dominantAxisOf` "must match exactly" discipline. `maxDirections=1` MUST reduce the selector to exactly today's `dominantAxis`.
- **Weight model (locked):** for a selected direction `d`, accumulation weight `w_d = viewFactor · (r_d / r_max)` where `r_d = |dot(n, axis_d)|^p` (axis_d is the sign-matched canonical axis, so `dot>0`), `r_max = max over selected r_d`, and `viewFactor = viewAngleWeight ? max(0, dot(n, normalize(cam - p))) : 1.0`. Quantize as `sumW += uint(w_d · SCALE)`, `sumDW += int(newValue · w_d · SCALE)`. `SCALE = 10000.0`.
- Namespace `Engine::Spatial`, no `vk` prefix. Architecture constants from `DirectionalTSDFTypes.h`.
- **Build/test env** (see memory `project-repo-build-worktree-gotchas`): `VULKAN_SDK=/usr/local`; submodules initialized; `example2` broken at HEAD (stub `example2/ShadowMap.cpp` `int main(){return 0;}` to configure); build ONLY `--target vkspatial_tests` (never the whole project); sources are GLOB'd → re-run `cmake -S . -B build` after adding a file; test binary `./build/test/vkspatial_tests`; device is Apple M4 Max (UMA). Keep GPU test sample counts well under ~1000 (Engine::Core large-N non-determinism, `docs/KNOWN_ISSUES_engine_core_large_n.md`).
- Commit after every task with a passing build + tests.

## File Structure

**New files:**
- `src/Engine/Spatial/DirectionalIntegrationQuality.h` — `IntegrationQuality` struct + shared `TopKDirections(n, q)` selector (header-only, so CPU and any host caller share one definition). Documents the exact arithmetic the shader must mirror.
- `test/test_directionalTSDFQuality.cpp` — quality unit tests + the interproximal single-vs-multi benchmark.

**Modified files:**
- `src/shader/directional_tsdf_integrate.comp` — replace single-direction write with a top-K loop + weight; add push-constant fields `g_maxDirections`, `g_dirExponent`, `g_viewAngleWeight`.
- `src/Engine/Spatial/DirectionalTSDF.h` / `.cpp` — hold an `IntegrationQuality m_quality` (settable), build the multi-direction write-set, pass the new push constants, and refine `mergeCandidates`.

---

## Task 1: `IntegrationQuality` + shared Top-K selector + multi-direction write-set

**Files:**
- Create: `src/Engine/Spatial/DirectionalIntegrationQuality.h`
- Modify: `src/Engine/Spatial/DirectionalTSDF.h` (add `IntegrationQuality m_quality;` + `void SetIntegrationQuality(const IntegrationQuality&)`), `src/Engine/Spatial/DirectionalTSDF.cpp` (write-set builder)
- Test: `test/test_directionalTSDFQuality.cpp`

**Interfaces:**
- Produces:
  - `struct Engine::Spatial::IntegrationQuality { uint32_t maxDirections = 1; uint32_t dirExponent = 4; bool viewAngleWeight = false; };`
  - `struct Engine::Spatial::DirWeight { uint8_t direction; float relWeight; };`
  - `// Up to q.maxDirections sign-matched canonical dirs with r_d/r_max >= 0.05, r_d = |n_axis|^q.dirExponent, sorted desc (tie x,y,z). relWeight = r_d/r_max. maxDirections==1 → exactly {dominantAxisOf(n), 1.0}.`
    `int Engine::Spatial::TopKDirections(const Eigen::Vector3f& n, const IntegrationQuality& q, DirWeight out[6]);` (returns count)
  - `void DirectionalTSDF::SetIntegrationQuality(const IntegrationQuality&);`

- [ ] **Step 1: Write the failing test**

`test/test_directionalTSDFQuality.cpp`:
```cpp
#include "Engine/Spatial/DirectionalIntegrationQuality.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial;

TEST(IntegrationQuality, K1MatchesDominantAxis) {
    IntegrationQuality q; // {1,4,false}
    DirWeight out[6];
    // +Z dominant
    int n = TopKDirections(Eigen::Vector3f(0.1f, -0.2f, 0.9f).normalized(), q, out);
    ASSERT_EQ(n, 1);
    EXPECT_EQ(out[0].direction, 4u); // +Z
    EXPECT_FLOAT_EQ(out[0].relWeight, 1.0f);
}

TEST(IntegrationQuality, AxisAlignedK2StillSingle) {
    IntegrationQuality q; q.maxDirections = 2;
    DirWeight out[6];
    int n = TopKDirections(Eigen::Vector3f(0, 0, 1), q, out); // pure +Z
    ASSERT_EQ(n, 1) << "secondary axes have r=0, excluded by minRelWeight";
    EXPECT_EQ(out[0].direction, 4u);
}

TEST(IntegrationQuality, DiagonalK2SplitsAcrossTwoLayers) {
    IntegrationQuality q; q.maxDirections = 2;
    DirWeight out[6];
    Eigen::Vector3f n(1, 0, 1); n.normalize(); // 45° between +X and +Z
    int cnt = TopKDirections(n, q, out);
    ASSERT_EQ(cnt, 2);
    EXPECT_EQ(out[0].relWeight, 1.0f);          // dominant normalized to 1
    EXPECT_NEAR(out[1].relWeight, 1.0f, 1e-4f); // equal split at exactly 45°
    // both are +X(0) and +Z(4)
    uint8_t a = out[0].direction, b = out[1].direction;
    EXPECT_TRUE((a == 0u && b == 4u) || (a == 4u && b == 0u));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests`
Expected: FAIL — `DirectionalIntegrationQuality.h` not found.

- [ ] **Step 3: Write `DirectionalIntegrationQuality.h`**

```cpp
#pragma once
#include <Eigen/Core>
#include <cstdint>

namespace Engine::Spatial {

    struct IntegrationQuality {
        uint32_t maxDirections = 1;   // K; 1 = single dominant (current behavior)
        uint32_t dirExponent = 4;     // p; integer, applied by repeated multiply
        bool viewAngleWeight = false; // multiply weight by max(0, dot(n, viewDir))
    };

    struct DirWeight { uint8_t direction; float relWeight; };

    inline float ipow(float x, uint32_t p) { float r = 1.0f; for (uint32_t i = 0; i < p; ++i) r *= x; return r; }

    // Sign-matched canonical dir per axis: +X=0,-X=1,+Y=2,-Y=3,+Z=4,-Z=5.
    // r_d = |n_axis|^p. Keep dirs with r_d/r_max >= 0.05, sorted desc, tie order x,y,z,
    // up to q.maxDirections. relWeight = r_d/r_max. Returns count (>=1).
    // maxDirections==1 reduces to exactly dominantAxisOf(n) with relWeight 1.
    inline int TopKDirections(const Eigen::Vector3f &n, const IntegrationQuality &q, DirWeight out[6]) {
        const float ax = std::abs(n.x()), ay = std::abs(n.y()), az = std::abs(n.z());
        const uint8_t dx = n.x() >= 0.0f ? 0u : 1u;
        const uint8_t dy = n.y() >= 0.0f ? 2u : 3u;
        const uint8_t dz = n.z() >= 0.0f ? 4u : 5u;
        // candidates in tie order x,y,z, sorted by |component| desc via 3 comparisons
        DirWeight c[3] = {{dx, ax}, {dy, ay}, {dz, az}};
        // stable sort desc by relWeight-field-as-|component| (only 3, keep x>y>z on ties)
        for (int i = 0; i < 3; ++i)
            for (int j = i + 1; j < 3; ++j)
                if (c[j].relWeight > c[i].relWeight) std::swap(c[i], c[j]);
        const float rmax = ipow(c[0].relWeight, q.dirExponent);
        int cnt = 0;
        const uint32_t K = q.maxDirections < 1 ? 1u : q.maxDirections;
        for (int i = 0; i < 3 && uint32_t(cnt) < K; ++i) {
            const float rd = ipow(c[i].relWeight, q.dirExponent);
            const float rel = rmax > 0.0f ? rd / rmax : 0.0f;
            if (i == 0) { out[cnt++] = {c[0].direction, 1.0f}; continue; }
            if (rel >= 0.05f) out[cnt++] = {c[i].direction, rel};
        }
        return cnt;
    }

} // namespace Engine::Spatial
```
(Add `#include <algorithm>`/`<cmath>` as needed.)

- [ ] **Step 4: Wire `IntegrationQuality` into `DirectionalTSDF` + the write-set**

In `DirectionalTSDF.h`: add member `IntegrationQuality m_quality;` and `void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }`.
In `DirectionalTSDF.cpp`, the IntegrationWriteSet loop (currently `const uint8_t d = dominantAxisOf(normals[i]); ... writeSet.insert({gx,gy,gz,d});`) becomes:
```cpp
        DirWeight dw[6];
        const int nd = TopKDirections(normals[i], m_quality, dw);
        // (inside the group-box loop, for each covered (gx,gy,gz):)
        for (int di = 0; di < nd; ++di)
            writeSet.insert({gx, gy, gz, dw[di].direction});
```
Keep everything else (box computation, halo) identical. `#include "Engine/Spatial/DirectionalIntegrationQuality.h"`.

- [ ] **Step 5: Build + run the new tests AND the existing suite (regression)**

Run: `cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests`
Run: `./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.*:*Directional*:*HostStore*'`
Expected: new `IntegrationQuality.*` PASS; all existing `*Directional*` PASS unchanged (default `m_quality` is `{1,4,false}` → `TopKDirections` returns exactly the dominant axis → write-set identical to before).

- [ ] **Step 6: Commit**
```bash
git add src/Engine/Spatial/DirectionalIntegrationQuality.h src/Engine/Spatial/DirectionalTSDF.h \
        src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDFQuality.cpp
git commit -m "feat(spatial): IntegrationQuality + shared top-K direction selector + multi-dir write-set"
```

---

## Task 2: Multi-direction + view-angle weighting in the integrate kernel

**Files:**
- Modify: `src/shader/directional_tsdf_integrate.comp`, `src/Engine/Spatial/DirectionalTSDF.cpp` (push constants)
- Test: `test/test_directionalTSDFQuality.cpp` (append)

**Interfaces:**
- Consumes: `IntegrationQuality` (Task 1); the kernel's push constant block gains `uint g_maxDirections; uint g_dirExponent; uint g_viewAngleWeight;`.
- Produces: integrate kernel that accumulates into up to K layers with weight `w_d = viewFactor · relWeight`.

- [ ] **Step 1: Write the failing test** — a 45° sample writes BOTH layers; view-angle attenuates grazing samples

Append to `test/test_directionalTSDFQuality.cpp` (uses `DebugDownloadGroupVoxels` to inspect accumulated `weight`=sumW/SCALE per layer):
```cpp
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"

TEST(IntegrationQuality, MultiDirectionWritesTwoLayers) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf; tsdf.Build(ctx);
    IntegrationQuality q; q.maxDirections = 2; tsdf.SetIntegrationQuality(q);
    // one sample near origin, normal at 45° between +X and +Z
    std::vector<Eigen::Vector3f> p{Eigen::Vector3f(0, 0, 0)};
    std::vector<Eigen::Vector3f> n{Eigen::Vector3f(1, 0, 1).normalized()};
    tsdf.Integrate(p, n, Eigen::Vector3f(0, 0, 5), Eigen::Vector3f::Zero());
    // the group containing the origin voxel should have nonzero weight in BOTH +X and +Z layers
    Eigen::Vector3i b = tsdf.LocalBase();
    // owner group of voxel (0,0,0): g=(0,0,0)
    auto gx = tsdf.DebugDownloadGroupVoxels({0,0,0,0}); // +X
    auto gz = tsdf.DebugDownloadGroupVoxels({0,0,0,4}); // +Z
    auto anyWeighted = [](const auto &grp){ for (auto &v : grp) if (v.weight > 0.0f) return true; return false; };
    EXPECT_TRUE(anyWeighted(gx)) << "+X layer got no contribution";
    EXPECT_TRUE(anyWeighted(gz)) << "+Z layer got no contribution";
}
```
(If the exact owner-group key math differs, compute it from `LocalBase()` + voxel→group as the existing tests do; keep the assertion "both layers nonzero".)

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.MultiDirectionWritesTwoLayers'`
Expected: FAIL — kernel still writes only the dominant layer (only one of +X/+Z is nonzero).

- [ ] **Step 3: Rewrite the integrate shader's write section**

In `directional_tsdf_integrate.comp`: add to the push-constant block `uint g_maxDirections; uint g_dirExponent; uint g_viewAngleWeight;`. Replace `uint dir = dominantAxis(...)` and the single accumulation with a shared top-K identical to CPU `TopKDirections`, and loop the band per selected direction:
```glsl
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
```
Then in `main`, after computing `rayDir`/`depth`:
```glsl
    vec3 nrm = vec3(s.nx, s.ny, s.nz);
    uint dirs[3]; float rel[3];
    int nd = topK(nrm, dirs, rel);
    float viewFactor = (g_viewAngleWeight != 0u) ? max(0.0, dot(nrm, -rayDir)) : 1.0;
```
and inside the band loop, replace the single-direction cell lookup + accumulation with a loop over `di in [0,nd)`:
```glsl
        for (int di = 0; di < nd; di++) {
            uint dir = dirs[di];
            uint cell = (((uint(lz)*uint(LOCAL_GRID)+uint(ly))*uint(LOCAL_GRID)+uint(lx))*6u) + dir;
            uint poolIndex = g_indexGrid[cell];
            if (poolIndex == 0xFFFFFFFFu) continue;
            ivec3 lv = v & 7;
            uint addr = poolIndex*512u + (uint(lv.z)*8u+uint(lv.y))*8u+uint(lv.x);
            float w = viewFactor * rel[di];
            atomicAdd(g_pool[addr].sumDW, int(newValue * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumW,  uint(w * TSDF_SCALE));
            atomicOr(g_meta[poolIndex].packed, 1u << 16);
        }
```
Keep `dominantAxis()` for reference but it is now unused (or delete it). **Backward-compat check:** with `g_maxDirections=1, g_viewAngleWeight=0`, `topK` returns `{dominant,1.0}` and `w = 1.0`, so `sumW += TSDF_SCALE` / `sumDW += newValue*TSDF_SCALE` — identical to the pre-change kernel.

- [ ] **Step 4: Pass the new push constants from `DirectionalTSDF.cpp`**

In the integrate dispatch, extend the push-constant struct with `m_quality.maxDirections`, `m_quality.dirExponent`, `uint(m_quality.viewAngleWeight ? 1 : 0)` (append after the existing camera fields; keep byte layout in sync with the shader block).

- [ ] **Step 5: Build + run new test + full regression**

Run: `cmake --build build --parallel --target vkspatial_tests`
Run: `./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.*:*Directional*:*HostStore*'`
Expected: `MultiDirectionWritesTwoLayers` PASS; all existing `*Directional*` PASS unchanged (default quality `{1,4,false}` reproduces prior kernel exactly — verify `SinglePlaneExtractsSurfacePoints`, `IntegrationWritesSignedBandAndMarksDirty`, `OpposingSurfacesRemainSeparate`, the Phase5 submit-count tests all still pass).

- [ ] **Step 6: Commit**
```bash
git add src/shader/directional_tsdf_integrate.comp src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDFQuality.cpp
git commit -m "feat(spatial): multi-direction + view-angle weighted TSDF integration (opt-in)"
```

---

## Task 3: Merge/split refinement — explicit 60° strong split

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp` (`mergeCandidates`)
- Test: `test/test_directionalTSDFQuality.cpp` (append)

**Interfaces:**
- Consumes: existing `mergeCandidates` (position 0.6·voxel + normal 30° cos>0.866 clustering).
- Produces: merge that additionally NEVER merges candidates whose normal is > 60° from a cluster mean, even if position-close (explicit strong split, spec §7 / doc §15).

**Note:** The current `mergeCandidates` already creates a new cluster when a candidate is outside the 30° merge cone, so >30° candidates already split. This task makes the 60° "strong split" an explicit, documented guard (and a regression anchor), and is deliberately small.

- [ ] **Step 1: Write the failing test**

Append:
```cpp
// Two candidates at the same voxel, normals 90° apart, must remain two points.
TEST(IntegrationQuality, StrongSplitKeepsPerpendicularSurfaces) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf; tsdf.Build(ctx);
    IntegrationQuality q; q.maxDirections = 2; tsdf.SetIntegrationQuality(q);
    // a thin corner: +X-facing and +Z-facing samples meeting near a shared voxel
    std::vector<Eigen::Vector3f> p, n;
    for (int i = -4; i <= 4; ++i) { p.emplace_back(0.0f, i*0.05f, 0.02f); n.emplace_back(0,0,1); }
    for (int i = -4; i <= 4; ++i) { p.emplace_back(0.02f, i*0.05f, 0.0f); n.emplace_back(1,0,0); }
    tsdf.Integrate(p, n, Eigen::Vector3f(1,0,1), Eigen::Vector3f::Zero());
    // expect points carrying BOTH a +Z-ish and +X-ish normal to survive (not merged into one blurred normal)
    bool hasZ=false, hasX=false;
    for (auto &pt : tsdf.PointCloud()) {
        if (pt.normal.z() > 0.7f) hasZ = true;
        if (pt.normal.x() > 0.7f) hasX = true;
    }
    EXPECT_TRUE(hasZ && hasX) << "perpendicular surfaces were merged into a blurred normal";
}
```

- [ ] **Step 2: Run to verify current behavior**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.StrongSplitKeepsPerpendicularSurfaces'`
Expected: Establish whether it already passes (same-direction-layer extraction likely already keeps them separate). If it PASSES already, that confirms the existing split suffices; still add the explicit guard in Step 3 as a regression anchor and keep the test. If it FAILS, Step 3 fixes it.

- [ ] **Step 3: Add the explicit strong-split guard**

In `mergeCandidates`, before the 30° merge test, add a hard reject at 60° so a candidate never merges into a cluster whose mean normal is more than 60° away:
```cpp
        const float strongSplitCos = 0.5f; // 60° — never merge beyond this (§7)
        ...
        for (auto &cl : clusters) {
            const Eigen::Vector3f mean = cl.posSum / float(cl.count);
            const Eigen::Vector3f meanN = cl.nSum.normalized();
            if (nrm.dot(meanN) < strongSplitCos) continue; // strong split: cannot merge
            if ((pos - mean).norm() < posThresh && nrm.dot(meanN) > cosThresh) {
                ... // existing weighted merge
            }
        }
```
(The existing `< kNumDirections` new-cluster path is unchanged.)

- [ ] **Step 4: Build + run + regression**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.*:*Directional*'`
Expected: `StrongSplitKeepsPerpendicularSurfaces` PASS; all existing `*Directional*` PASS unchanged (`OpposingSurfacesRemainSeparate` in particular).

- [ ] **Step 5: Commit**
```bash
git add src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDFQuality.cpp
git commit -m "feat(spatial): explicit 60-degree strong split in candidate merge"
```

---

## Task 4: Interproximal quality benchmark — single vs multi-direction

**Files:**
- Test: `test/test_directionalTSDFQuality.cpp` (append)

**Interfaces:**
- Consumes: `DirectionalTSDF` + `IntegrationQuality`. Proves the quality claim quantitatively.

**Contract:** On a synthetic thin-wall / interproximal scene (two near-parallel surfaces ~1 voxel apart, with slightly off-axis normals so a single dominant axis under-covers the boundary), multi-direction (`K=2`, `viewAngleWeight=true`) yields **at least as complete** a surface as single-direction (`K=1`) — measured by extracted-point coverage of the reference band — and no worse RMSE. This is the demonstrable "innovative quality" result.

- [ ] **Step 1: Write the benchmark test**

Append a test that builds two `DirectionalTSDF` instances over the SAME synthetic interproximal scan (a scripted multi-frame or single-frame set of samples on two thin walls with normals tilted ~20–30° off-axis), one with `{1,4,false}` and one with `{2,4,true}`, then compares:
```cpp
TEST(IntegrationQuality, MultiDirectionImprovesInterproximalCoverage) {
    auto run = [](IntegrationQuality q) {
        Engine::Core::Context ctx;
        DirectionalTSDF tsdf; tsdf.Build(ctx); tsdf.SetIntegrationQuality(q);
        std::vector<Eigen::Vector3f> p, n;
        // two thin walls ~1 voxel apart, normals tilted off +Z toward +X (interproximal-like)
        for (int i = -6; i <= 6; ++i) for (int j = -6; j <= 6; ++j) {
            Eigen::Vector3f na(0.35f, 0.0f, 0.94f); na.normalize();
            p.emplace_back(i*0.05f, j*0.05f, 0.0f);  n.push_back(na);
            Eigen::Vector3f nb(-0.35f, 0.0f, 0.94f); nb.normalize();
            p.emplace_back(i*0.05f, j*0.05f, 0.12f); n.push_back(nb);
        }
        tsdf.Integrate(p, n, Eigen::Vector3f(0,0,5), Eigen::Vector3f::Zero());
        return tsdf.PointCloud().size();
    };
    size_t single = run(IntegrationQuality{1,4,false});
    size_t multi  = run(IntegrationQuality{2,4,true});
    EXPECT_GE(multi, single) << "multi-direction should cover at least as much surface";
    // and it should produce a meaningfully non-empty reconstruction
    EXPECT_GT(multi, 0u);
}
```
(Tune the tilt/spacing so the assertion is meaningful — the off-axis normals must make the single-dominant layer under-cover the boundary while multi-direction fills it. Keep total samples < 400 to stay under the large-N threshold. If `>=` proves too weak to be interesting, strengthen to a coverage ratio, but never assert a flaky exact count.)

- [ ] **Step 2: Build + run**

Run: `cmake --build build --parallel --target vkspatial_tests && ./build/test/vkspatial_tests --gtest_filter='IntegrationQuality.MultiDirectionImprovesInterproximalCoverage'`
Expected: PASS (multi ≥ single, multi > 0).

- [ ] **Step 3: Full-suite regression**

Run: `./build/test/vkspatial_tests`
Expected: all PASS (the 1 pre-existing `EngineWideBVHTest.RadiusMatchesCpu` skip is fine); every existing `*Directional*` unchanged.

- [ ] **Step 4: Commit**
```bash
git add test/test_directionalTSDFQuality.cpp
git commit -m "test(spatial): interproximal benchmark shows multi-direction coverage win"
```

---

## Out of scope (follow-on)
- Flipping the default `IntegrationQuality` to `{2,4,true}` (and updating the few tests that assert single-direction accumulation) — a later decision once the benchmark is trusted.
- Confidence terms beyond view-angle (frontend confidence / distance weight) — no frontend exists in this repo.
- 8B fixed-point unification (spec §2) and UMA co-refinement (spec §8) — separate plans.

## Self-Review Notes
- **Spec §7 coverage:** multi-direction soft integration → Tasks 1–2; confidence (view-angle) fusion → Task 2; merge/split → Task 3; demonstrable quality → Task 4. Confidence's frontend/direction terms beyond view-angle are deferred (no frontend input).
- **Backward-compat:** every task's regression step re-runs `*Directional*`; defaults `{1,4,false}` make `TopKDirections`/kernel/merge reproduce today's behavior (Task 1 `K1MatchesDominantAxis` + Task 2 backward-compat check pin this).
- **CPU/GPU match:** the selector arithmetic (integer `ipow`, tie order x,y,z, `minRelWeight=0.05`, relWeight=r_d/r_max) is specified once and mirrored verbatim in `TopKDirections` (Task 1) and `topK` (Task 2) — the plan's highest-risk correctness point.
- **Pinned open values:** K=2, p=4, minRelWeight=0.05, strongSplitCos=0.5 (60°), posThresh=0.6·voxel, mergeCos=0.866 (30°). All tunable; documented in `IntegrationQuality`/`mergeCandidates`.
