# Compact Best-TSDF v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `CompactDirectionalTSDF` (flat-hash, ~28× leaner than block DirectionalTSDF) to block-grade accuracy by porting **stored-gradient (mode-3)** and adding **point-to-plane** integration — lowering extraction RMSE (flat near-perfect, edges better) at a fraction of block memory.

**Architecture:** Expand the compact `DirEntry` from 16B to 24B (reclaim the unused `pad`, add 3 gradient accumulators `sumNx/y/z`). The integrate shader accumulates `sumN = Σ n·w·SCALE` and gains a point-to-plane SDF toggle. The extract shader keeps its legacy zero-crossing *position* but takes the *normal* from `normalize(sumN)` (mode-3 hybrid), falling back to the existing central-difference gradient when `sumN` is degenerate. Validated by A/B on `tsdf_benchmark`'s analytic cube/cylinder fixtures.

**Tech Stack:** C++17, Vulkan compute (MoltenVK/M4 Max UMA), GLSL (glslc), GoogleTest (`vkspatial_tests`), fixed-point accumulation (`TSDF_SCALE = 10000`).

## Global Constraints

- **Branch:** work on `main` (the compact path is main-only). Start from a clean tree — note the working tree currently has an *uncommitted, unrelated* A1 experiment on the **block** path (`SetPointToPlane` in `DirectionalTSDF.*` + `directional_tsdf_integrate.comp` + `tsdf_benchmark.cpp`); do NOT revert or depend on it, and `git add` only the files each task names.
- **DirEntry layout (binding):** `{ uint key; int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; }` = **24 bytes**, std430 stride 24 (all 4-byte scalars). The C++ struct and BOTH shader mirrors (`compact_directional_integrate.comp`, `compact_directional_extract.comp`) must match this layout exactly. Field *names* may differ per shader (extract uses `weightedDistanceSum`/`weightSum`) but order/types/size must not.
- **Fixed-point:** `TSDF_SCALE = 10000`; `sumN` accumulates the raw (un-normalized) observed normal `n·w·SCALE`, normalized only at extraction (`normalize(sumN)`), scale cancels.
- **Point-to-plane sign:** must match the projective form (positive in front / camera side). `sdf_p2p = dot(vCenter - p, normalize(nrm))`.
- **Extract:** compact extraction is mode-3 hybrid only (legacy zero-crossing position + stored-gradient normal); central-difference normal remains as the degenerate fallback. No new extract-mode API.
- **Movable window unchanged:** 32-bit key / 512³ window is untouched by the 24B change.
- **Build:** `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j`. **Run:** `./build/test/vkspatial_tests --gtest_filter='<Suite.Name>'`. Shader compile check: `cd src/shader && glslc -fshader-stage=comp -I. <file> -o /tmp/x.spv`.
- **Commits:** repo convention commits per task; follow the user's git policy for when they actually land.

---

## File Structure

- **Modify** `src/Engine/Spatial/CompactDirectionalTSDF.h` — `DirEntry` → 24B + `static_assert`; `CompactEntry` gains `normal`; add `SetPointToPlane(bool)`; declare member.
- **Modify** `src/Engine/Spatial/CompactDirectionalTSDF.cpp` — `IntegratePC` gains `pointToPlane`; fill it; `Reset()` 6-field init; `DownloadEntries` unpacks `sumN`→normal.
- **Modify** `src/shader/compact_directional_integrate.comp` — `DirEntry` 24B; PC `g_pointToPlane`; point-to-plane sdf; accumulate `sumN`.
- **Modify** `src/shader/compact_directional_extract.comp` — `DirEntry` 24B; normal from `normalize(sumN)`, central-difference fallback.
- **Modify** `test/test_compactDirectional.cpp` — layout, gradient accumulation, mode-3 normal, point-to-plane sign tests.
- **Modify** `example2/tsdf_benchmark.cpp` — expose compact point-to-plane for the A/B measurement (Task 4).

---

### Task 1: DirEntry → 24B (C++ + both shaders)

**Files:**
- Modify: `src/Engine/Spatial/CompactDirectionalTSDF.h` (DirEntry struct + asserts)
- Modify: `src/Engine/Spatial/CompactDirectionalTSDF.cpp` (`Reset()` init)
- Modify: `src/shader/compact_directional_integrate.comp:24`, `src/shader/compact_directional_extract.comp` (DirEntry struct)
- Test: `test/test_compactDirectional.cpp`

**Interfaces:**
- Produces: `struct DirEntry { uint32_t key; int32_t sumDW; uint32_t sumW; int32_t sumNx, sumNy, sumNz; }` (24B) used by all later tasks.

- [ ] **Step 1: Write the failing test** — append to `test/test_compactDirectional.cpp`:

```cpp
TEST(CompactDirectional, DirEntryIs24Bytes) {
    using Engine::Spatial::DirEntry;
    static_assert(sizeof(DirEntry) == 24, "DirEntry must be 24B (key+sumDW+sumW+sumN)");
    EXPECT_EQ(offsetof(DirEntry, sumW), 8u);
    EXPECT_EQ(offsetof(DirEntry, sumNx), 12u);
    EXPECT_EQ(offsetof(DirEntry, sumNz), 20u);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j`
Expected: FAIL — `sizeof(DirEntry)` is 16; `sumNx` not a member.

- [ ] **Step 3: Expand the C++ struct** in `CompactDirectionalTSDF.h` — replace the `DirEntry` definition:

```cpp
    struct DirEntry {
        uint32_t key;   // packDirKey(voxel, dir); 0xFFFFFFFF = empty
        int32_t sumDW;  // sum(value_i * w_i) * 10000
        uint32_t sumW;  // sum(w_i)          * 10000
        int32_t sumNx;  // sum(n_i * w_i)    * 10000  (stored gradient; was pad)
        int32_t sumNy;
        int32_t sumNz;
    }; // 24 bytes. Layout must match DirEntry in compact_directional_{integrate,extract}.comp.
    static_assert(sizeof(DirEntry) == 24, "DirEntry must be 24 bytes");
    static_assert(offsetof(DirEntry, sumNx) == 12);
```

(Ensure `<cstddef>` is included for `offsetof`; add it if missing.)

- [ ] **Step 4: Fix `Reset()`** in `CompactDirectionalTSDF.cpp` — the empty-fill initializer must list 6 fields:

```cpp
        std::vector<DirEntry> empty(m_hashCapacity, {EMPTY_KEY, 0, 0u, 0, 0, 0});
```

- [ ] **Step 5: Expand both shader mirrors.** In `src/shader/compact_directional_integrate.comp` replace line 24:

```glsl
struct DirEntry { uint key; int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; };
```

In `src/shader/compact_directional_extract.comp` replace its `struct DirEntry { ... };` with (keeping the extract file's field names):

```glsl
struct DirEntry {
    uint key;
    int  weightedDistanceSum;  // sum of (signed distance * weight)
    uint weightSum;            // sum of weights
    int  sumNx;                // stored gradient (Σ n·w·SCALE)
    int  sumNy;
    int  sumNz;
};
```

- [ ] **Step 6: Verify shaders compile**

Run: `cd src/shader && glslc -fshader-stage=comp -I. compact_directional_integrate.comp -o /tmp/i.spv && glslc -fshader-stage=comp -I. compact_directional_extract.comp -o /tmp/e.spv && echo SHADERS_OK`
Expected: `SHADERS_OK`

- [ ] **Step 7: Run tests (layout + no regression)**

Run: `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='CompactDirectional*'`
Expected: PASS — new layout test passes; existing compact tests still pass (buffer alloc uses `sizeof(DirEntry)`, so 24B is absorbed; gradient unused yet).

- [ ] **Step 8: Commit**

```bash
git add src/Engine/Spatial/CompactDirectionalTSDF.h src/Engine/Spatial/CompactDirectionalTSDF.cpp \
        src/shader/compact_directional_integrate.comp src/shader/compact_directional_extract.comp \
        test/test_compactDirectional.cpp
git commit -m "feat(spatial): compact DirEntry 24B (reclaim pad for stored-gradient sumN)"
```

---

### Task 2: Integrate — accumulate sumN + point-to-plane toggle

**Files:**
- Modify: `src/shader/compact_directional_integrate.comp` (PC, sdf, sumN accumulate)
- Modify: `src/Engine/Spatial/CompactDirectionalTSDF.h` (`SetPointToPlane` + member; `CompactEntry.normal`)
- Modify: `src/Engine/Spatial/CompactDirectionalTSDF.cpp` (`IntegratePC` + fill; `DownloadEntries` normal)
- Test: `test/test_compactDirectional.cpp`

**Interfaces:**
- Consumes: `DirEntry` 24B (Task 1).
- Produces: `void CompactDirectionalTSDF::SetPointToPlane(bool)`; `CompactEntry{ ...; Eigen::Vector3f normal; }` populated from `normalize(sumN)`.

- [ ] **Step 1: Write the failing test** — append to `test/test_compactDirectional.cpp`:

```cpp
TEST(CompactDirectional, IntegrateAccumulatesStoredNormal) {
    Engine::Core::Context ctx;
    Engine::Spatial::CompactDirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.05f, 0.15f);           // voxel, truncation
    tsdf.SetIntegrationQuality({1, 4, true}); // single dominant dir, view weight on

    // A planar patch at z=0 with +Z normals, camera on +Z looking down.
    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -6; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j) { pts.emplace_back(i*0.02f, j*0.02f, 0.0f); nrm.emplace_back(0,0,1); }
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1));

    auto entries = tsdf.DownloadEntries();
    ASSERT_GT(entries.size(), 0u);
    int checked = 0;
    for (const auto& e : entries) {
        if (e.weight <= 0.0f) continue;
        EXPECT_NEAR(e.normal.z(), 1.0f, 1e-2f);
        EXPECT_NEAR(e.normal.x(), 0.0f, 1e-2f);
        ++checked;
    }
    EXPECT_GT(checked, 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='CompactDirectional.IntegrateAccumulatesStoredNormal'`
Expected: FAIL — `CompactEntry` has no `normal` (compile error), or normal is zero.

- [ ] **Step 3: Add `normal` to `CompactEntry` + `SetPointToPlane`** in `CompactDirectionalTSDF.h`:

```cpp
    struct CompactEntry {
        Eigen::Vector3f center;
        uint32_t direction;
        float tsdf;
        float weight;
        Eigen::Vector3f normal;   // v1: normalize(sumN); zero if degenerate
    };
```
Add near the other setters:
```cpp
        // v1: integrate SDF form. true = point-to-plane (removes grazing bias, default),
        // false = projective ray distance. Flows into integrate PC g_pointToPlane.
        void SetPointToPlane(bool on) { m_pointToPlane = on; }
```
Add member (near `m_quality`):
```cpp
        bool m_pointToPlane = true; // v1 default: point-to-plane
```

- [ ] **Step 4: Extend `IntegratePC` + fill + `DownloadEntries`** in `CompactDirectionalTSDF.cpp`.

In the `IntegratePC` struct, append after `viewAngleWeight`:
```cpp
            uint32_t pointToPlane;
```
Wait — the compact `IntegratePC` ends with `originX/Y/Z`. Insert `pointToPlane` as the LAST field (after `originZ`), and mirror it as the last PC field in the shader (Step 6). Update the `ipc{...}` initializer to pass `uint32_t(m_pointToPlane ? 1u : 0u)` as the final argument.

In `DownloadEntries`, after computing `ce.tsdf`/`ce.weight`, add:
```cpp
            Eigen::Vector3f sumN(float(e.sumNx), float(e.sumNy), float(e.sumNz));
            float nlen = sumN.norm();
            ce.normal = nlen > 1e-6f ? (sumN / nlen) : Eigen::Vector3f::Zero();
```

- [ ] **Step 5: Shader — PC field, point-to-plane sdf, accumulate sumN** in `compact_directional_integrate.comp`.

Add to the PC block (after `g_originZ`):
```glsl
    uint  g_pointToPlane;   // v1: 1 = point-to-plane, 0 = projective
```
Replace the `voxel2point` line (currently `float voxel2point = depth - dot(vCenter - cam, rayDir);`):
```glsl
        float voxel2point = (g_pointToPlane != 0u)
                                ? dot(vCenter - p, normalize(nrm))   // point-to-plane
                                : depth - dot(vCenter - cam, rayDir); // projective
```
After the two existing `atomicAdd(g_hash[slot].sumDW ...)` / `sumW` lines, add:
```glsl
            atomicAdd(g_hash[slot].sumNx, int(nrm.x * w * TSDF_SCALE));
            atomicAdd(g_hash[slot].sumNy, int(nrm.y * w * TSDF_SCALE));
            atomicAdd(g_hash[slot].sumNz, int(nrm.z * w * TSDF_SCALE));
```

- [ ] **Step 6: Verify shader compiles + run test**

Run: `cd src/shader && glslc -fshader-stage=comp -I. compact_directional_integrate.comp -o /tmp/i.spv && echo OK && cd ../.. && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='CompactDirectional.IntegrateAccumulatesStoredNormal'`
Expected: `OK` then PASS (stored normal ≈ +Z).

- [ ] **Step 7: Commit**

```bash
git add src/shader/compact_directional_integrate.comp src/Engine/Spatial/CompactDirectionalTSDF.h \
        src/Engine/Spatial/CompactDirectionalTSDF.cpp test/test_compactDirectional.cpp
git commit -m "feat(spatial): compact integrate accumulates sumN + point-to-plane toggle"
```

---

### Task 3: Extract — mode-3 stored-gradient normal

**Files:**
- Modify: `src/shader/compact_directional_extract.comp` (normal from `normalize(sumN)`, central-diff fallback)
- Test: `test/test_compactDirectional.cpp`

**Interfaces:**
- Consumes: `DirEntry.sumNx/y/z` (Task 1/2), `estimateNormal` (existing central-difference, kept as fallback).

- [ ] **Step 1: Write the failing test** — append to `test/test_compactDirectional.cpp`:

```cpp
TEST(CompactDirectional, ExtractUsesStoredGradientNormal) {
    Engine::Core::Context ctx;
    Engine::Spatial::CompactDirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.05f, 0.15f);
    tsdf.SetIntegrationQuality({1, 4, true});

    std::vector<Eigen::Vector3f> pts, nrm;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) { pts.emplace_back(i*0.02f, j*0.02f, 0.0f); nrm.emplace_back(0,0,1); }
    tsdf.Integrate(pts, nrm, Eigen::Vector3f(0, 0, 1));

    auto cloud = tsdf.ExtractPointCloud(1u << 18, /*merge=*/false);
    ASSERT_GT(cloud.normals.size(), 0u);
    double meanNz = 0.0;
    for (const auto& n : cloud.normals) meanNz += n.z();
    meanNz /= double(cloud.normals.size());
    EXPECT_GT(meanNz, 0.99); // stored-gradient normals ~ +Z, denoised
}
```

- [ ] **Step 2: Run to verify it fails or is noisy**

Run: `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='CompactDirectional.ExtractUsesStoredGradientNormal'`
Expected: FAIL or marginal — the current extract computes the normal by central differences (`estimateNormal`), which is noisier than the stored gradient; meanNz may be < 0.99.

- [ ] **Step 3: Use the stored gradient for the normal** in `compact_directional_extract.comp`. In `main()`, replace the `estimateNormal(...)` call block that fills `surfaceNormal` with:

```glsl
    // mode-3 hybrid: stored-gradient normal (denoised), central-difference fallback.
    vec3 sumN = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
    float sumNlen = length(sumN);
    vec3 surfaceNormal;
    if (sumNlen > 1e-6) {
        surfaceNormal = sumN / sumNlen;
    } else if (!estimateNormal(voxel, direction, centerValue, surfaceNormal)) {
        return; // degenerate: no stored gradient and no finite-difference gradient
    }
```

(Keep `estimateCrossingPosition` for the position — position is unchanged. Keep the `estimateNormal` function definition; it is now the fallback.)

- [ ] **Step 4: Verify shader compiles + run test**

Run: `cd src/shader && glslc -fshader-stage=comp -I. compact_directional_extract.comp -o /tmp/e.spv && echo OK && cd ../.. && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='CompactDirectional*'`
Expected: `OK` then PASS (all compact tests, including the new stored-gradient normal).

- [ ] **Step 5: Commit**

```bash
git add src/shader/compact_directional_extract.comp test/test_compactDirectional.cpp
git commit -m "feat(spatial): compact extract uses stored-gradient normal (mode-3 hybrid)"
```

---

### Task 4: Benchmark A/B — RMSE + memory validation

**Files:**
- Modify: `example2/tsdf_benchmark.cpp` (expose compact point-to-plane on the Compact-Directional path)
- Run only + record numbers.

**Interfaces:**
- Consumes: `CompactDirectionalTSDF::SetPointToPlane` (Task 2).

- [ ] **Step 1: Wire the compact instance to honor the existing `--p2p` flag.** In `example2/tsdf_benchmark.cpp`, find where the `CompactDirectionalTSDF cd;` is built (`cd.Build(ctx, voxel, kTruncation);`) and add, immediately after Build:

```cpp
        cd.SetPointToPlane(g_p2p);
```

(`g_p2p` is the file-scope flag already added by the uncommitted block-path A1 experiment; if it is not present in the tree, add `bool g_p2p = false;` in the anonymous namespace and parse `--p2p` in `main` as `else if (a == "--p2p") { g_p2p = true; }`.)

- [ ] **Step 2: Build the harness**

Run: `VULKAN_SDK=/usr/local cmake --build build --target tsdf_benchmark -j`
Expected: links OK.

- [ ] **Step 3: Measure the compact path A/B (projective vs point-to-plane)**

Run:
```
./build/example2/tsdf_benchmark --shape both        2>&1 | grep -E "Compact|insight \((cube|cylinder), COMBINE"
./build/example2/tsdf_benchmark --shape both --p2p  2>&1 | grep -E "Compact|insight \((cube|cylinder), COMBINE"
```
Expected: with `--p2p`, the Compact-Directional flat/edge error drops sharply (flat → near-zero on cube), matching the block-path A1 result (mean −45~62%, flat near-perfect), while memory (`Compact-Directional(fine)` KB) is unchanged (~28× below block). Record both tables.

- [ ] **Step 4: Record results** — append a "Compact Best-TSDF v1 — measured (2026-07-27)" subsection to `docs/superpowers/specs/2026-07-27-compact-best-tsdf-v1-design.md` with: compact RMSE/edge/flat (projective vs p2p), compact vs block memory (KB and ×), and the stored-gradient normal result. State whether the v1 gate (RMSE ≤ block mode-3+p2p, ~28× less memory) is met.

- [ ] **Step 5: Commit**

```bash
git add example2/tsdf_benchmark.cpp docs/superpowers/specs/2026-07-27-compact-best-tsdf-v1-design.md
git commit -m "feat(bench)+docs: measure Compact Best-TSDF v1 (point-to-plane + stored-gradient)"
```

---

## Self-Review

**1. Spec coverage** (against `2026-07-27-compact-best-tsdf-v1-design.md`):
- §2 DirEntry 24B → Task 1. ✓
- §3 integrate sumN + point-to-plane → Task 2. ✓
- §4 extract mode-3 stored-gradient normal → Task 3. ✓
- §6 benchmark RMSE + memory validation → Task 4. ✓
- Deferred (variance-adaptive, E2 quantization, streaming) — correctly out of scope. ✓

**2. Placeholder scan:** every step has exact code/commands + expected output. No TBD/"handle errors". ✓

**3. Type consistency:** `DirEntry{key,sumDW,sumW,sumNx,sumNy,sumNz}` (24B) identical across C++ + both shaders; `SetPointToPlane`, `g_pointToPlane`, `CompactEntry.normal`, `sumN`/`normalize(sumN)` used consistently across Tasks 1-4. ✓

**4. Ambiguity:** point-to-plane sign fixed (`dot(vCenter - p, normalize(nrm))`, positive in front); extract keeps legacy position, replaces only the normal; `pointToPlane` is the LAST PC field in both shader and C++. ✓
