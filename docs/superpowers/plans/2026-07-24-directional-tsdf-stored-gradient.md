# Stored-Gradient DirectionalTSDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accumulate observed surface normals per DirectionalTSDF voxel so extraction uses a denoised normal (`normalize(sumN)`) and projects position onto the isosurface along it (`center − c·truncation·n̂`), measured 3-way (legacy / FD / stored) via the existing headless `--dump` harness.

**Architecture:** Expand `GpuTsdfVoxel` from 8B to 20B with three int accumulators (`sumNx/y/z`). The integrate kernel adds `atomicAdd` of the sample normal (already in hand) into those fields. The extract kernel gains a `g_mode==2` path that reads `sumN`, uses `normalize(sumN)` as the normal, and projects the point to `center − c·truncation·normalize(sumN)`. This is measure-first GPU-only: `HostTsdfVoxel` stays 8B (gradient not persisted across eviction), which is correct for the single-window feature-compare experiment.

**Tech Stack:** C++17, Vulkan compute (GLSL 460, integer atomics), Eigen, GoogleTest, CMake. `Engine::Spatial::DirectionalTSDF`.

## Global Constraints

- Measure-first GPU-only: `HostTsdfVoxel` stays 8B; gradient lives only in the GPU pool for the frame. Do NOT change host-store/wire/writeback persistence.
- Default behavior unchanged: `SetExtractMode` defaults to `0` (legacy); modes 0 and 1 (legacy, FD-projection) must be byte-identical to before this plan. The stored path is mode 2 only.
- Layout coherence: exactly two shaders address pool voxels — `directional_tsdf_integrate.comp` and `directional_tsdf_extract.comp`. Both must declare `GpuVoxel` as the SAME 20-byte layout `{int sumDW; uint sumW; int sumNx; int sumNy; int sumNz;}`. (`register_reusable` binds only `PoolIndexList`; `classify` does not touch the pool — do not change them.)
- Accumulation scale: reuse `TSDF_SCALE` (=10000.0) for the normal accumulators; the scale cancels in `normalize(sumN)`, so only overflow matters (safe: `|sumNx| ≤ sumW`-magnitude, within int32).
- Position units: stored TSDF `c = sumDW/sumW` is truncation-normalized; metric distance is `c·truncation`; the outward surface normal aligns with `∇sdf`, so the isosurface point is `center − c·truncation·n̂`.
- Primary oracle is the DETERMINISTIC cube (`--shape cube`); it has `flat`+`edge` regions (no curved). The cylinder has curved but the branch base (`c4cb890`) lacks the large-N fix (`ddee668` on main), so cylinder `--dump` jitters ±1-2 points — judge cylinder on means, cube on exact values.
- Every git commit message ends with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Stage ONLY the files each task names. NEVER `git add -A`/`.` — the tree has unrelated untracked `.ply` files, a `docs/*.pdf`, and a `lib/SPIRV-Reflect` submodule change that must not be committed.

## File Structure

- `src/Engine/Spatial/DirectionalTSDFTypes.h` (modify) — `GpuTsdfVoxel` → 20B + static_asserts.
- `src/shader/directional_tsdf_integrate.comp` (modify) — `GpuVoxel` 20B; accumulate `sumN`.
- `src/shader/directional_tsdf_extract.comp` (modify) — `GpuVoxel` 20B; `g_refine`→`g_mode`, add `g_truncation`, mode-2 path.
- `src/Engine/Spatial/StreamingResidencyBackend.cpp` (modify) — zero the 3 new fields on upload.
- `src/Engine/Spatial/DirectionalTSDF.{h,cpp}` (modify) — `SetExtractMode(uint32_t)`, `ExtractPC` mode+truncation, dispatch.
- `example2/tsdf_feature_compare.cpp` (modify) — rename API calls (Task 3), add stored 3rd column (Task 4).

---

### Task 1: Expand voxel layout to 20B (behavioral no-op)

**Files:**
- Modify: `src/Engine/Spatial/DirectionalTSDFTypes.h:60-63` (struct), `:107-109` (asserts)
- Modify: `src/shader/directional_tsdf_integrate.comp:13` (`GpuVoxel`)
- Modify: `src/shader/directional_tsdf_extract.comp:11` (`GpuVoxel`)
- Modify: `src/Engine/Spatial/StreamingResidencyBackend.cpp:253-256` (upload), `:13` (comment)

**Interfaces:**
- Consumes: nothing new.
- Produces: `GpuTsdfVoxel` with public fields `int32_t sumDW, sumNx, sumNy, sumNz; uint32_t sumW;` totalling 20 bytes, layout `{sumDW@0, sumW@4, sumNx@8, sumNy@12, sumNz@16}`. Shaders mirror it as `struct GpuVoxel { int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; };`.

- [ ] **Step 1: Capture the legacy `--dump` baseline (regression oracle)**

```bash
cmake -S . -B build
cmake --build build --target tsdf_feature_compare -j
APP=$(find build -type f -name tsdf_feature_compare | head -1)
"$APP" --shape cube --dump | tee /tmp/sg_baseline_cube.txt
```
Expected: the current legacy/FD table prints, exit 0. Cube is deterministic — keep this file for Step 6.

- [ ] **Step 2: Expand the C++ struct + asserts**

In `src/Engine/Spatial/DirectionalTSDFTypes.h`, replace the `GpuTsdfVoxel` struct (lines 60-63):

```cpp
    struct GpuTsdfVoxel {
        int32_t sumDW = 0;
        uint32_t sumW = 0;
        int32_t sumNx = 0; // Σ n·w·TSDF_SCALE per direction layer (measure-first: not persisted)
        int32_t sumNy = 0;
        int32_t sumNz = 0;
    }; // 20B
```

Replace the two asserts (lines 108-109) with:

```cpp
    static_assert(sizeof(GpuTsdfVoxel) == 20);
    static_assert(offsetof(GpuTsdfVoxel, sumW) == 4);
    static_assert(offsetof(GpuTsdfVoxel, sumNx) == 8);
    static_assert(offsetof(GpuTsdfVoxel, sumNz) == 16);
```

- [ ] **Step 3: Mirror the 20B layout in both pool shaders**

In `src/shader/directional_tsdf_integrate.comp` line 13, replace:

```glsl
struct GpuVoxel { int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; };
```

In `src/shader/directional_tsdf_extract.comp` line 11, replace:

```glsl
struct GpuVoxel { int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; };
```

- [ ] **Step 4: Zero the new fields on Streaming upload**

In `src/Engine/Spatial/StreamingResidencyBackend.cpp`, the upload loop constructs each `GpuTsdfVoxel` field-by-field into mapped staging (lines 254-256). Mapped memory is not default-constructed, so the new fields are garbage unless set. After the `g.sumDW = ...;` line add:

```cpp
                g.sumNx = 0;
                g.sumNy = 0;
                g.sumNz = 0;
```

Also update the stale byte-count comment on line 13 (`// 4096` → `// 10240`).

- [ ] **Step 5: Build**

```bash
cmake --build build --target vkspatial_tests tsdf_feature_compare -j
```
Expected: builds clean (asserts pass → `sizeof==20` confirmed at compile time; shaders compile).

- [ ] **Step 6: Verify the layout change is a behavioral no-op**

```bash
APP=$(find build -type f -name tsdf_feature_compare | head -1)
diff <("$APP" --shape cube --dump) /tmp/sg_baseline_cube.txt && echo "CUBE UNCHANGED"
```
Expected: empty diff, `CUBE UNCHANGED`. The 20B layout with unused/zeroed gradient fields must not alter any extracted number. If it differs, a shader `GpuVoxel` stride or the upload zero-fill is wrong.

- [ ] **Step 7: Commit**

```bash
git add src/Engine/Spatial/DirectionalTSDFTypes.h src/shader/directional_tsdf_integrate.comp src/shader/directional_tsdf_extract.comp src/Engine/Spatial/StreamingResidencyBackend.cpp
git commit -m "$(cat <<'EOF'
feat(spatial): expand GpuTsdfVoxel to 20B with unused gradient accumulators

Behavioral no-op: adds sumNx/y/z (zeroed on upload), used by later tasks.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Accumulate observed normals in the integrate kernel

**Files:**
- Modify: `src/shader/directional_tsdf_integrate.comp:113-115` (add three `atomicAdd`)

**Interfaces:**
- Consumes: the 20B `GpuVoxel` from Task 1; `nrm` (sample normal, line 81); `w` (line 112).
- Produces: `g_pool[addr].sumN{x,y,z}` populated with `Σ round(n·w·TSDF_SCALE)`. Consumed by Task 3's extract mode 2.

- [ ] **Step 1: Add the normal accumulation**

In `src/shader/directional_tsdf_integrate.comp`, immediately after the two existing accumulators (lines 113-114, `atomicAdd(...sumDW...)` and `atomicAdd(...sumW...)`) and before `atomicOr(g_meta...)` (line 115), insert:

```glsl
            atomicAdd(g_pool[addr].sumNx, int(nrm.x * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumNy, int(nrm.y * w * TSDF_SCALE));
            atomicAdd(g_pool[addr].sumNz, int(nrm.z * w * TSDF_SCALE));
```

- [ ] **Step 2: Build**

```bash
cmake --build build --target tsdf_feature_compare -j
```
Expected: builds clean.

- [ ] **Step 3: Verify no regression on modes 0/1 (they ignore sumN)**

```bash
APP=$(find build -type f -name tsdf_feature_compare | head -1)
diff <("$APP" --shape cube --dump) /tmp/sg_baseline_cube.txt && echo "CUBE UNCHANGED"
```
Expected: empty diff, `CUBE UNCHANGED`. Accumulating `sumN` cannot affect legacy/FD extraction (they never read it); this confirms the accumulation didn't perturb the existing paths. (`sumN` is validated for correctness in Task 4, where mode 2 consumes it.)

- [ ] **Step 4: Commit**

```bash
git add src/shader/directional_tsdf_integrate.comp
git commit -m "$(cat <<'EOF'
feat(spatial): accumulate observed normals per voxel in integrate kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Extract mode 2 + SetExtractMode API (harness rename-only)

**Files:**
- Modify: `src/shader/directional_tsdf_extract.comp` — PC block, position/normal computation
- Modify: `src/Engine/Spatial/DirectionalTSDF.h:73` (setter), `:133` (member)
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp:34-42` (`ExtractPC`), `:286-288` (dispatch)
- Modify: `example2/tsdf_feature_compare.cpp:139,154` (rename `SetSubvoxelRefine`→`SetExtractMode`)

**Interfaces:**
- Consumes: `sumN` from Task 2; `g_truncation` (new push constant).
- Produces: `void DirectionalTSDF::SetExtractMode(uint32_t mode)` (0=legacy, 1=FD, 2=stored; default 0), member `uint32_t m_extractMode = 0;`. Shader push constant renamed `g_refine`→`g_mode` (uint) plus new trailing `float g_truncation`. `ExtractPC{...; uint32_t mode; float truncation;}`. Task 4 calls `SetExtractMode(2)`.

- [ ] **Step 1: Extend the extract push constant and rename the mode field**

In `src/shader/directional_tsdf_extract.comp`, replace the `PC` block's trailing `uint g_refine;` line so the block ends:

```glsl
    int   g_baseZ;
    uint  g_mode;        // 0 = legacy average, 1 = FD projection, 2 = stored-gradient
    float g_truncation;  // metric = c * truncation (mode 2 position magnitude)
};
```

- [ ] **Step 2: Compute position AND normal per mode**

In `src/shader/directional_tsdf_extract.comp`, the current code computes `vec3 pos` in a `g_refine==1u` if/else and then a single `vec3 n = grad / len;`. Replace that whole block (the `vec3 pos; if (g_refine == 1u) {...} else {...}` and the following `vec3 n = grad / len;`) with:

```glsl
    vec3 pos;
    vec3 n;
    if (g_mode == 2u) {
        // Stored-gradient: denoised normal from accumulated observed normals, position
        // projected onto the isosurface along it. Fall back if the accumulator is degenerate.
        vec3 sumN = vec3(float(center.sumNx), float(center.sumNy), float(center.sumNz));
        float sl = length(sumN);
        if (sl < 1e-6) {
            pos = posSum / float(crossings);
            n = grad / len;
        } else {
            vec3 nhat = sumN / sl;
            vec3 voxelCenter = (vec3(v) + vec3(0.5)) * g_voxelSize;
            pos = voxelCenter - c * g_truncation * nhat;
            n = nhat;
        }
    } else if (g_mode == 1u) {
        vec3 voxelCenter = (vec3(v) + vec3(0.5)) * g_voxelSize;
        vec3 stepv = c * g_voxelSize * grad / dot(grad, grad);
        pos = (length(stepv) > g_voxelSize) ? (posSum / float(crossings)) : (voxelCenter - stepv);
        n = grad / len;
    } else {
        pos = posSum / float(crossings);
        n = grad / len;
    }
```

- [ ] **Step 3: Rename the C++ setter/member and extend `ExtractPC`**

In `src/Engine/Spatial/DirectionalTSDF.h`, replace line 73:

```cpp
        // Extraction mode: 0 = legacy axis-crossing average, 1 = FD gradient projection,
        // 2 = stored-gradient (normalize(sumN) normal + isosurface projection). Default 0.
        void SetExtractMode(uint32_t mode) { m_extractMode = mode; }
```

and replace the member on line 133:

```cpp
        uint32_t m_extractMode = 0;
```

In `src/Engine/Spatial/DirectionalTSDF.cpp`, replace `ExtractPC` (lines 34-42):

```cpp
        struct ExtractPC {
            uint32_t numGroups;
            float voxelSize;
            uint32_t maxCandidates;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
            uint32_t mode;
            float truncation;
        };
```

and the dispatch initializer (lines 286-288):

```cpp
            ExtractPC epc{uint32_t(groupSlots.size()), m_voxelSize, m_maxCandidates,
                          localBase.x(), localBase.y(), localBase.z(),
                          m_extractMode, m_truncation};
```

- [ ] **Step 4: Keep the harness compiling (rename calls only, no new column yet)**

In `example2/tsdf_feature_compare.cpp`, replace the two calls: line 139 `dirLegacy.SetSubvoxelRefine(false);` → `dirLegacy.SetExtractMode(0);` and line 154 `dirRefined.SetSubvoxelRefine(true);` → `dirRefined.SetExtractMode(1);`. Leave everything else in this file for Task 4.

- [ ] **Step 5: Build**

```bash
cmake --build build --target vkspatial_tests tsdf_feature_compare -j
```
Expected: builds clean (PC layout: shader 8 scalar fields ↔ `ExtractPC` 8 fields, both `mode` then `truncation` last).

- [ ] **Step 6: Verify modes 0/1 unchanged**

```bash
APP=$(find build -type f -name tsdf_feature_compare | head -1)
diff <("$APP" --shape cube --dump) /tmp/sg_baseline_cube.txt && echo "CUBE UNCHANGED"
```
Expected: empty diff, `CUBE UNCHANGED`. The harness still runs modes 0 (legacy) and 1 (FD); the new mode-2 code and `g_truncation` must not alter their output.

- [ ] **Step 7: Commit**

```bash
git add src/shader/directional_tsdf_extract.comp src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp example2/tsdf_feature_compare.cpp
git commit -m "$(cat <<'EOF'
feat(spatial): extract mode 2 (stored-gradient) + SetExtractMode API

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Harness 3-way (add Stored column) + measure the win

**Files:**
- Modify: `example2/tsdf_feature_compare.cpp` — `CompareResult` (104-117), `RunCompare` (135-167), `PrintReport` (172-206)

**Interfaces:**
- Consumes: `DirectionalTSDF::SetExtractMode(2)` (Task 3).
- Produces: the deliverable measurement — a 3-way `--dump` table (legacy / FD / Stored) for position and normal-angle error.

- [ ] **Step 1: Add stored fields to `CompareResult`**

In `example2/tsdf_feature_compare.cpp`, after the refined fields in `CompareResult` (after line 116 `dirNormStatsRefined{};`), add:

```cpp
        std::vector<Eigen::Vector3f> dirPtsStored;         // mode 2 positions
        std::vector<Eigen::Vector3f> dirNormalsStored;     // mode 2 normals
        std::array<RegionStat, kNumRegions> dirStatsStored{};     // stored position error
        std::array<RegionStat, kNumRegions> dirNormStatsStored{}; // stored normal-angle error
```

- [ ] **Step 2: Run the stored instance in `RunCompare`**

In `RunCompare`, after the `dirRefined` block that ends at line 160 (the `for (const auto &e : dirRefined.PointCloud())` loop) and before the `accumulate(...)` calls, insert:

```cpp
        // ---- DirectionalTSDF, stored-gradient extraction (mode 2) ----
        Engine::Spatial::DirectionalTSDF dirStored;
        dirStored.Build(ctx, voxel, truncation);
        dirStored.SetIntegrationQuality({3, 4, true});
        dirStored.SetExtractMode(2);
        for (const auto &v : r.views)
            dirStored.Integrate(v.points, v.normals, v.camPos, Eigen::Vector3f::Zero());
        for (const auto &e : dirStored.PointCloud()) {
            r.dirPtsStored.push_back(e.position);
            r.dirNormalsStored.push_back(e.normal);
        }
```

Then, after the existing `accumulateNormals(...r.dirNormStatsRefined)` call (line 166), add:

```cpp
        accumulate(shape, voxel, r.dirPtsStored, r.dirStatsStored);
        accumulateNormals(shape, voxel, r.dirPtsStored, r.dirNormalsStored, r.dirNormStatsStored);
```

- [ ] **Step 3: Add the Stored columns to `PrintReport`**

In `PrintReport`, extend the header count line (line 176-178) to include the stored count, and add a `Stored.*` column pair to each of the two tables. Replace the position-table header + row `printf`s (lines 181-192) with:

```cpp
        std::printf("[position error mm]\n");
        std::printf("%-7s | %11s %11s | %11s %11s | %11s %11s | %11s %11s\n", "region",
                    "Simple.mean", "Simple.max", "Dir.mean", "Dir.max", "FD.mean", "FD.max",
                    "Stored.mean", "Stored.max");
        std::printf("--------+-------------------------+-------------------------"
                    "+-------------------------+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.simpleStats[reg].count == 0 && r.dirStats[reg].count == 0 &&
                r.dirStatsRefined[reg].count == 0 && r.dirStatsStored[reg].count == 0)
                continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f\n",
                        regionName(reg), r.simpleStats[reg].mean(), r.simpleStats[reg].maxErr,
                        r.dirStats[reg].mean(), r.dirStats[reg].maxErr,
                        r.dirStatsRefined[reg].mean(), r.dirStatsRefined[reg].maxErr,
                        r.dirStatsStored[reg].mean(), r.dirStatsStored[reg].maxErr);
        }
```

Replace the normal-angle-table header + row `printf`s (lines 195-204) with:

```cpp
        std::printf("[normal-angle error deg]  (edge = reference only, normal is discontinuous)\n");
        std::printf("%-7s | %11s %11s | %11s %11s | %11s %11s\n", "region", "Dir.mean", "Dir.max",
                    "FD.mean", "FD.max", "Stored.mean", "Stored.max");
        std::printf("--------+-------------------------+-------------------------"
                    "+-------------------------\n");
        for (int reg = 0; reg < kNumRegions; ++reg) {
            if (r.dirNormStats[reg].count == 0 && r.dirNormStatsRefined[reg].count == 0 &&
                r.dirNormStatsStored[reg].count == 0)
                continue;
            std::printf("%-7s | %9.4f %11.4f | %9.4f %11.4f | %9.4f %11.4f\n", regionName(reg),
                        r.dirNormStats[reg].mean(), r.dirNormStats[reg].maxErr,
                        r.dirNormStatsRefined[reg].mean(), r.dirNormStatsRefined[reg].maxErr,
                        r.dirNormStatsStored[reg].mean(), r.dirNormStatsStored[reg].maxErr);
        }
```

- [ ] **Step 4: Build**

```bash
cmake --build build --target tsdf_feature_compare -j
```
Expected: builds clean.

- [ ] **Step 5: Measure — the acceptance gate (deterministic cube)**

```bash
APP=$(find build -type f -name tsdf_feature_compare | head -1)
"$APP" --shape cube --dump
"$APP" --shape cylinder --dump
```
Expected (acceptance):
- **Headline (cube, deterministic):** in `[normal-angle error deg]`, the `flat` row shows **`Stored.mean` ≪ `FD.mean` (≈1.8773°) and ≪ `Dir.mean`** — the accumulated normal is far more accurate than the finite-difference one. This is the primary must-pass signal.
- **Position (cube, deterministic):** `flat` `Stored.mean` ≤ `Dir.mean` (≈0.0110) — reverses the lite (FD) regression (0.0115). At minimum no regression vs legacy.
- **Cylinder:** `curved` `Stored` normal-angle and position improve vs Dir/FD (judge on means; ±1-2 pt jitter is the documented pre-existing nondeterminism, not a failure).
- If cube `flat` normal-angle Stored is NOT clearly better than FD, the accumulation (Task 2), the `normalize(sumN)` normal, or the upload zero-fill is wrong — STOP and report the numbers rather than committing.

- [ ] **Step 6: Commit**

```bash
git add example2/tsdf_feature_compare.cpp
git commit -m "$(cat <<'EOF'
feat(feature-compare): 3-way legacy/FD/stored --dump; stored-gradient wins normals

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Spec §1 (GpuTsdfVoxel 20B + asserts) → Task 1 Step 2. ✓
- Spec §2 (integrate sumN accumulation, TSDF_SCALE) → Task 2 Step 1. ✓
- Spec §3 (extract g_mode + g_truncation + mode-2 normal/position + fallback) → Task 3 Steps 1-2. ✓
- Spec §4 (Streaming upload zero-fill; Unified no-change; layout audit integrate+extract only) → Task 1 Step 4 + Global Constraints. ✓
- Spec §5 (SetExtractMode + ExtractPC mode/truncation + dispatch) → Task 3 Step 3. ✓
- Spec §6 (harness 3-way, keep refined=FD, add stored, relabel DirRef→FD) → Task 3 Step 4 (rename) + Task 4. ✓
- Spec §검증 (cube bit-identical regression each step; headline cube flat normal-angle Stored≪FD; cylinder means) → Task 1/2/3 Step 6 + Task 4 Step 5. ✓
- Spec 범위 밖 (host-store persistence, thin structures, mesh, compression) → not implemented, by design. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `GpuTsdfVoxel` (C++) and `GpuVoxel` (GLSL, both shaders) all `{sumDW, sumW, sumNx, sumNy, sumNz}`, 20B, identical order — Task 1. `SetExtractMode(uint32_t)`/`m_extractMode` defined Task 3 Step 3, used Task 3 Step 4 and Task 4 Step 2. `ExtractPC{...,mode,truncation}` (C++) ↔ `g_mode`,`g_truncation` (GLSL) same trailing order — Task 3 Steps 1,3. `CompareResult` stored fields (`dirPtsStored`, `dirNormalsStored`, `dirStatsStored`, `dirNormStatsStored`) defined Task 4 Step 1, used Steps 2-3. `TSDF_SCALE` reused for accumulation (Task 2) matches `kTsdfFixedScale` used by the backend conversion. ✓
