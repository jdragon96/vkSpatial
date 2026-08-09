# ICP Registration-Quality Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise local ICP alignment quality via four stacked improvements — sub-voxel target, robust correspondences, coarse-to-fine annealing, and a motion model — each proven by a deterministic perturbation-recovery harness plus a live residual RMSE.

**Architecture:** Measurement is built first (residual RMSE in `RegistrationResult` + a `Track()`-driven harness reusing `Engine::Eval::RmseMetrics`), then three quality tiers applied to BOTH the GPU path (`GpuPointToPlaneIcp` + `icp_iterate.comp.glsl` + `GpuIcpTracker`) and the CPU path (`AlignPointToPlaneIcp` + `PointToPlaneIcpTracker`), plus a shared constant-velocity motion model in `RegistrationThread`. The GPU and CPU residual/weight formulas stay identical so `GpuIcp.SolveMatchesCpuOnCorner` keeps them in lockstep.

**Tech Stack:** C++17, Vulkan compute (MoltenVK), GLSL, Eigen, GoogleTest.

## Global Constraints

- **MoltenVK has no GPU float atomics** — every new GPU reduction value uses the existing per-workgroup `shared int` fixed-point pattern (`SCALE = 10000.0`, `atomicAdd`, one 32-bit slot per quantity, CPU sums the per-workgroup partials as `double`).
- **GPU and CPU paths must stay numerically consistent** — identical residual, robust-weight, normal-rejection, and annealing formulas on both sides; `GpuIcp.SolveMatchesCpuOnCorner` must stay within its tolerance after every task.
- **Preserve the per-solve grid hoist** — `prepareCentred` builds the grid + uploads buffers ONCE per `Solve`; annealing must NOT reintroduce a per-iteration rebuild (see Task 5: build the grid at the coarsest cell once, shrink only the per-iteration distance filter passed as a push constant).
- **Full descriptive names, no abbreviations** in all new/edited code (`maxCorrespondenceDistance`, `sumOfSquaredResiduals`, `sourceNormal`, `truncationDistance`, `huberScale`, `numberOfWorkgroups`) — matches the user's standing preference.
- **Reuse `Engine::Eval::RmseMetrics`** (`src/Engine/Eval/RmseMetrics.h`: `NearestNeighbourRMSE`, `AccuracyRMSE`) for reconstruction-vs-ground-truth RMSE; do not reinvent it.
- **Acceptance gate every task:** the three targets build (`cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8`), the suite stays green (**≈238 pass / 1 skip / 0 fail**, `./build/test/vkspatial_tests`), and — for the quality tiers — the harness's recovered-pose error and RMSE **improve or hold** versus the previous tier's recorded baseline.
- **Branch:** work is currently on `main`; execution (subagent-driven) must first move to a feature branch (setup step of that skill). Do NOT push to origin. Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Untouchable working-tree WIP:** `src/Engine/Spatial/AdvancedTSDF.cpp` and `src/shader/advanced_tsdf_{compact,integrate}.comp.glsl` are the user's edits — never stage/revert them; stage only each task's own files (`git add <paths>`, never `git add .`).
- Shaders compile at runtime from `src/shader/` (no CMake change for `.glsl`); CMake globs sources + tests (reconfigure `cmake -S . -B build` picks up new files).

---

## File Structure

| File | Change |
|---|---|
| `src/Engine/Pipeline/Registration/RegistrationTypes.h` | add `float rmse` to `RegistrationResult` (Task 1) |
| `src/shader/icp_iterate.comp.glsl` | 28→29 slot reduction (Σ residual², Task 1); source-normals binding + Huber weight + normal rejection (Task 4); per-iteration distance filter (Task 5) |
| `src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.{h,cpp}` | `IterOut.sumOfSquaredResiduals` + 29-slot readback + `res.rmse` (Task 1); source-normals buffer/binding (Task 4); coarsest-cell grid + per-iteration `currentMaxCorrespondenceDistance` (Task 5) |
| `src/Engine/Pipeline/Registration/PointToPlaneIcp.h` | CPU Σ residual² + `res.rmse` (Task 1); Huber + normal rejection (Task 4); per-iteration distance filter (Task 5) |
| `src/Engine/Pipeline/Types.h` | add `float truncationDistance` to `ModelSnapshot` (Task 3) |
| `src/Engine/Pipeline/Integration/IntegrationThread.cpp` | populate `snap.truncationDistance` (Task 3) |
| `src/Engine/Pipeline/Registration/Tracker.cpp` | sub-voxel target in both trackers (Task 3); pass source normals to `Solve` (Task 4) |
| `src/Engine/Pipeline/Registration/RegistrationThread.{h,cpp}` | constant-velocity motion prior + running RMSE accessor (Task 2 accessor, Task 5 motion) |
| `test/test_gpuIcp.cpp` | RMSE consistency test (Task 1); perturbation-recovery harness (Task 2); per-tier assertions (Tasks 3–5) |
| `example2/VoxelFillRenderStrategy.cpp` | live tracking RMSE in the stats panel (Task 2) |

---

## Task 1 — Residual RMSE in `RegistrationResult` (GPU + CPU)

**Files:** Modify `RegistrationTypes.h`, `icp_iterate.comp.glsl`, `GpuPointToPlaneIcp.{h,cpp}`, `PointToPlaneIcp.h`; Test `test/test_gpuIcp.cpp`.

**Interfaces produced:** `Engine::Registration::RegistrationResult::rmse` (float, `sqrt(sumOfSquaredResiduals / numberOfInliers)` of the final iteration's point-to-plane residuals; `0` when no inliers).

- [ ] **Step 1: Failing test — GPU and CPU RMSE agree and are non-zero**

Append to `test/test_gpuIcp.cpp` (uses the existing corner fixture helpers already in that file):

```cpp
TEST(GpuIcp, ResidualRmseMatchesCpu) {
    Engine::Core::Context context;
    Engine::Registration::PointCloud target = MakeCornerTarget(); // existing helper in this file
    std::vector<Eigen::Vector3f> source = PerturbedSource(target, 0.02f, 0.03f); // existing helper
    Engine::Registration::RegistrationParam params;
    params.maxCorrDist = 0.1f;

    const auto cpu = Engine::Registration::AlignPointToPlaneIcp(source, target,
                                                               Eigen::Matrix4f::Identity(), params);
    Engine::Pipeline::GpuPointToPlaneIcp gpu(context);
    const auto gpuResult = gpu.Solve(source, target, Eigen::Matrix4f::Identity(), params);

    EXPECT_GT(cpu.rmse, 0.0f);
    EXPECT_GT(gpuResult.rmse, 0.0f);
    EXPECT_NEAR(gpuResult.rmse, cpu.rmse, 1e-3f) << "gpu " << gpuResult.rmse << " cpu " << cpu.rmse;
}
```

(If the fixture helper names differ, reuse whatever `SolveMatchesCpuOnCorner` already uses in this file — do not invent new fixtures.)

- [ ] **Step 2: Run — fails to compile (`rmse` not a member)**

`cmake -S . -B build && cmake --build build --target vkspatial_tests -j8 2>&1 | grep -E "error:|no member named 'rmse'"`

- [ ] **Step 3: Add the field**

In `RegistrationTypes.h`'s `RegistrationResult`, add after `fitness`:
```cpp
        float rmse = 0.0f; // sqrt(mean squared point-to-plane residual) over inliers, final iteration; 0 if none
```

- [ ] **Step 4: GPU — add the Σ residual² accumulator (28 → 29 slots)**

In `icp_iterate.comp.glsl`:
- `shared int s_acc[28];` → `shared int s_acc[29];`
- zero + writeback guards `if (tid < 28u)` → `if (tid < 29u)` (both occurrences), and the partial stride `gl_WorkGroupID.x * 28u` → `* 29u`; the comment `[numWG * 28]` → `[numWG * 29]`.
- inside `if (best >= 0) { ... }`, after `atomicAdd(s_acc[27], 1);` add:
```glsl
            atomicAdd(s_acc[28], int(round(e * e * SCALE))); // sum of squared residuals (fixed-point)
```
(`e*e` is tiny in the centred frame — well within the int32 bound documented at the top of the file.)

- [ ] **Step 5: GPU host — read slot 28, compute rmse**

In `GpuPointToPlaneIcp.h`, add to `struct IterOut`:
```cpp
        double sumOfSquaredResiduals = 0.0;
```
In `GpuPointToPlaneIcp.cpp`:
- everywhere the partials buffer is sized/read as `* 28u`, change to `* 29u` (`prepareCentred`'s `AllocateHostVisibleReadback`, `dispatchCentred`'s `InvalidateMapped` + the `double acc[28]` → `acc[29]` + the `for (int k = 0; k < 28; ...)` → `< 29`).
- in `dispatchCentred`, after reading `out.inliers`, add:
```cpp
        out.sumOfSquaredResiduals = acc[28] / double(kScale);
```
- in `Solve`, capture the last iteration's value and set `res.rmse`. Inside the loop after `res.numInliers = size_t(a.inliers);` add `lastSumOfSquaredResiduals = a.sumOfSquaredResiduals;` (declare `double lastSumOfSquaredResiduals = 0.0;` before the loop), and after the loop before `res.valid`:
```cpp
        if (res.numInliers > 0)
            res.rmse = float(std::sqrt(lastSumOfSquaredResiduals / double(res.numInliers)));
```
(`#include <cmath>` is already present.)

- [ ] **Step 6: CPU — accumulate Σ residual² and set rmse**

In `PointToPlaneIcp.h`'s `AlignPointToPlaneIcp`, inside the iteration loop add `float sumOfSquaredResiduals = 0.0f;` next to `int inliers = 0;`, and in the correspondence loop after `++inliers;` add `sumOfSquaredResiduals += e * e;`. After `res.fitness = ...` in the loop, add:
```cpp
            res.rmse = inliers > 0 ? std::sqrt(sumOfSquaredResiduals / float(inliers)) : 0.0f;
```
(`#include <cmath>` is already present.)

- [ ] **Step 7: Run the test + full suite**

`cmake --build build --target vkspatial_tests -j8 && ./build/test/vkspatial_tests --gtest_filter='GpuIcp.*:LocalGrid.*:Pipeline.*'`
Expected: `GpuIcp.ResidualRmseMatchesCpu` PASS, all others still green.

- [ ] **Step 8: Commit** — `git add src/Engine/Pipeline/Registration/RegistrationTypes.h src/shader/icp_iterate.comp.glsl src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.h src/Engine/Pipeline/Registration/GpuPointToPlaneIcp.cpp src/Engine/Pipeline/Registration/PointToPlaneIcp.h test/test_gpuIcp.cpp` then commit `feat(icp): residual RMSE in RegistrationResult (GPU 29th fixed-point slot + CPU)`.

---

## Task 2 — Perturbation-recovery harness + live RMSE (baseline capture)

**Files:** Modify `RegistrationThread.{h,cpp}`, `example2/VoxelFillRenderStrategy.cpp`; Test `test/test_gpuIcp.cpp` (harness). Consumes `Engine::Eval::RmseMetrics`.

**Interfaces produced:** a gated harness test `GpuIcp.DISABLED_RegistrationQualityHarness` printing `recoveredTranslationError`, `recoveredRotationErrorRadians`, `Engine::Eval::NearestNeighbourRMSE(alignedSource, trueSurface)`, and residual `rmse`; `RegistrationThread::TrackerRmseAvg()`.

- [ ] **Step 1: Add `truncationDistance` to `ModelSnapshot` (the harness needs it; Task 3 consumes it)**

In `Types.h` `ModelSnapshot`, after `float voxel`, add `float truncationDistance = 0.0f;`. (Populating it in the live `IntegrationThread` pipeline is done in Task 3; the harness sets it directly on its hand-built snapshot.)

- [ ] **Step 2: Write the harness (gated, prints the metrics)**

Append to `test/test_gpuIcp.cpp`. It builds a **coarse-voxel-quantized** corner model whose entries carry the sub-voxel `tsdf`, so later tiers are measurable:

```cpp
TEST(GpuIcp, DISABLED_RegistrationQualityHarness) {
    const float voxel = 0.05f, truncation = 0.15f;
    // true surface points (sub-voxel), + a ModelSnapshot whose entries are the SAME surface snapped to
    // the voxel grid but carrying tsdf/normal so center - tsdf*truncation*normal recovers the surface.
    std::vector<Eigen::Vector3f> trueSurface = MakeCornerSurfacePoints(); // dense corner, sub-voxel
    Engine::Pipeline::ModelSnapshot model = QuantizeToModel(trueSurface, voxel, truncation); // helper (this task)
    model.voxel = voxel;
    model.truncationDistance = truncation;

    Eigen::Isometry3f knownPerturbation = Eigen::Isometry3f::Identity();
    knownPerturbation.translate(Eigen::Vector3f(0.02f, -0.015f, 0.01f));
    knownPerturbation.rotate(Eigen::AngleAxisf(0.03f, Eigen::Vector3f::UnitZ()));
    // trueNormals[i] is the (unit) surface normal at trueSurface[i]; the sensor frame sees both the
    // points and normals rotated by the perturbation (normals rotate, do not translate).
    std::vector<Eigen::Vector3f> trueNormals = MakeCornerSurfaceNormals(); // parallel to trueSurface
    Engine::Pipeline::Frame frame;
    for (size_t i = 0; i < trueSurface.size(); ++i) {
        frame.pts.push_back(knownPerturbation * trueSurface[i]);
        frame.nrm.push_back(knownPerturbation.rotation() * trueNormals[i]);
    }

    auto tracker = Engine::Pipeline::TrackerRegistry::Default().Create("icp");
    const auto result = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());

    const Eigen::Isometry3f error = result.pose * knownPerturbation; // should be ~identity
    const float recoveredTranslationError = error.translation().norm();
    const float recoveredRotationErrorRadians = Eigen::AngleAxisf(error.rotation()).angle();
    std::vector<Eigen::Vector3f> alignedSource;
    for (const auto &p : frame.pts) alignedSource.push_back(result.pose * p);
    const float reconRmse = Engine::Eval::NearestNeighbourRMSE(alignedSource, trueSurface);
    std::printf("[harness] transErr %.5f rotErr %.5f reconNnRmse %.5f residualRmse %.5f inliers %zu\n",
                recoveredTranslationError, recoveredRotationErrorRadians, reconRmse, result.rmse, result.inliers);
    SUCCEED();
}
```

Implement the small helpers (`MakeCornerSurfacePoints`, `QuantizeToModel` producing `AdvancedEntry`s with `center` snapped + `tsdf`/`normal` set, `frame.nrm`) in the test file. `QuantizeToModel` sets `entry.tsdf = signedDistanceOf(center)/truncation` so `center - tsdf*truncation*normal` returns the true surface — this is what Task 3 will exploit. Include `Engine/Eval/RmseMetrics.h`.

- [ ] **Step 2: Run it, RECORD the baseline numbers** in the plan's results note (these are the reference every later tier must beat/hold):

`./build/test/vkspatial_tests --gtest_also_run_disabled_tests --gtest_filter='GpuIcp.DISABLED_RegistrationQualityHarness'`

- [ ] **Step 3: Live RMSE — running mean in RegistrationThread**

In `RegistrationThread.h` add `util::RunningMean m_trackerRmse;` and `double TrackerRmseAvg() const { return m_trackerRmse.Mean(); }`. In `RegistrationThread.cpp::Run`, after a valid track, feed it: inside `if (a.valid)` add `m_trackerRmse.Add(a.rmse);` (mirror how `m_trackerMs` is used).

- [ ] **Step 4: Show it in the viewer stats panel**

In `example2/VoxelFillRenderStrategy.cpp`, where the pipeline-stage averages are drawn, add a line showing the tracking RMSE. If the strategy can reach the pipeline's `RegistrationThread` timing it already shows, surface `TrackerRmseAvg()` the same way (add a pipeline accessor if needed, mirroring the existing timing accessor path). Keep it one text line: `"ICP rmse (avg): %.4f"`.

- [ ] **Step 5: Build all three targets + suite green**

`cmake -S . -B build && cmake --build build --target vkspatial_tests voxel_fill_debugger registration_chair_demo -j8` then `./build/test/vkspatial_tests` (≈238/1/0; the harness stays disabled in the normal run).

- [ ] **Step 6: Commit** — stage the harness + RegistrationThread + VoxelFillRenderStrategy files; `feat(icp): perturbation-recovery harness + live tracking RMSE (baseline)`.

---

## Task 3 — Tier 1: sub-voxel target from stored `tsdf`

**Files:** Modify `Types.h` (`ModelSnapshot`), `IntegrationThread.cpp`, `Tracker.cpp` (both trackers); Test `test/test_gpuIcp.cpp`.

**Interfaces:** consumes `ModelSnapshot.truncationDistance` (added here). Produces sub-voxel target points in both trackers.

- [ ] **Step 1: Extend the harness assertion (failing)** — add to the harness (or a non-disabled variant) an assertion that sub-voxel alignment beats the Task-2 baseline, e.g. `EXPECT_LT(recoveredTranslationError, <recorded baseline>)`. It fails now (trackers still use raw centers).

- [ ] **Step 2: Populate `truncationDistance` in the live pipeline**

The `truncationDistance` field was added to `ModelSnapshot` in Task 2. Here, populate it for the real pipeline: in `IntegrationThread.cpp::buildSnapshot`, next to `snap.voxel = baseVoxel;` add `snap.truncationDistance = truncation;` (thread the map truncation into `buildSnapshot` the same way `baseVoxel` is passed — the call site has `m_cfg.truncation`).

- [ ] **Step 3: Project centers to the sub-voxel surface in both trackers**

In `Tracker.cpp`, in BOTH `PointToPlaneIcpTracker::Track` and `GpuIcpTracker::Track`, when building the target, replace `target.points.push_back(entry.center);` with the sub-voxel surface point:

```cpp
                    const float truncationDistance = model->truncationDistance > 0.0f
                                                         ? model->truncationDistance
                                                         : 0.0f;
                    const Eigen::Vector3f surfacePoint =
                            entry.center - entry.tsdf * truncationDistance * entry.normal;
                    target.points.push_back(surfacePoint);
                    target.normals.push_back(entry.normal);
```

**Verify the sign.** `tsdf` is normalized (`voxel2point/truncateDistance` in the integrate shader). The harness's `recoveredTranslationError` decides: if the sub-voxel projection makes it WORSE than baseline, flip the sign to `entry.center + entry.tsdf * truncationDistance * entry.normal` and re-measure. Whichever direction reduces the error is correct; keep that one. (For the GPU tracker this is CPU-side target construction — no shader change.)

- [ ] **Step 4: Build + harness + suite**

Build the three targets; run the harness (record the improved numbers); `./build/test/vkspatial_tests` green. The harness `transErr`/`reconNnRmse` must drop vs the Task-2 baseline.

- [ ] **Step 5: Commit** — `feat(icp): Tier 1 sub-voxel target via stored tsdf (both trackers)`.

---

## Task 4 — Tier 2: robust correspondences (Huber + normal rejection + tighter gate)

**Files:** Modify `icp_iterate.comp.glsl`, `GpuPointToPlaneIcp.{h,cpp}`, `Tracker.cpp` (pass source normals), `PointToPlaneIcp.h`; Test `test/test_gpuIcp.cpp`.

**Interfaces:** `GpuPointToPlaneIcp::Solve` gains a `const std::vector<Eigen::Vector3f> &sourceNormals` parameter; `AlignPointToPlaneIcp` gains the same. A `huberScale` + `normalCompatibilityCosine` on `RegistrationParam`.

- [ ] **Step 1: Failing test** — extend the harness to a case with outliers/noise where robust weighting must beat Tier 1 (`EXPECT_LT(recoveredTranslationError, <Tier1 baseline>)` on a noisier fixture).

- [ ] **Step 2: Add tuning fields to `RegistrationParam`**

In `RegistrationTypes.h` `RegistrationParam`, add:
```cpp
        float huberScale = 0.05f;               // robust-weight knee (world units; caller sets ~voxel)
        float normalCompatibilityCosine = 0.5f; // reject correspondence if sourceN·targetN < this (~60deg)
```
Tighten the default gate note: callers already set `maxCorrDist = 2*voxel`; Task 5 anneals it. (Leave the default value; the tracker sets it per voxel.)

- [ ] **Step 3: CPU — Huber weight + normal rejection**

In `AlignPointToPlaneIcp`, the signature gains `const std::vector<Eigen::Vector3f> &sourceNormals` (may be empty ⇒ skip the normal check). In the correspondence loop, after computing `q`,`n`,`e`:
```cpp
                if (!sourceNormals.empty()) {
                    const Eigen::Vector3f transformedSourceNormal = R * sourceNormals[sourceIndex];
                    if (transformedSourceNormal.dot(n) < params.normalCompatibilityCosine) continue;
                }
                const float absoluteResidual = std::abs(e);
                const float robustWeight = absoluteResidual <= params.huberScale
                                               ? 1.0f
                                               : params.huberScale / absoluteResidual;
                H += robustWeight * (J * J.transpose());
                b += robustWeight * (-J * e);
                sumOfSquaredResiduals += e * e; // RMSE stays unweighted (a true fit metric)
                ++inliers;
```
(Track `sourceIndex` with an indexed loop instead of range-for.)

- [ ] **Step 4: GPU — upload source normals, add binding, Huber + rejection in the shader**

- `GpuPointToPlaneIcp`: add a `m_sourceNormals` buffer; `Solve`/`prepareCentred` take `sourceNormals`, upload them with `writeVec3Buf` (centring does NOT apply to normals — upload as-is), and `Bind(6, *m_sourceNormals)`.
- Shader: add `layout(std430, binding=6) readonly buffer SrcNrm { vec4 g_srcNrm[]; };` and `float g_huberScale, g_normalCompatibilityCosine;` to the push-constant block (append after `g_maxCorr`; keep the C++ `IcpPC` in the SAME order and add the two floats). In `main`, transform the source normal by the pose's rotation (`mat3(g_T) * g_srcNrm[i].xyz`), and inside `if (best >= 0)`:
```glsl
            vec3 transformedSourceNormal = mat3(g_T) * g_srcNrm[i].xyz;
            if (dot(transformedSourceNormal, n) < g_normalCompatibilityCosine) { /* skip: fall through */ }
            else {
                float e = dot(p - q, n);
                float absoluteResidual = abs(e);
                float robustWeight = absoluteResidual <= g_huberScale ? 1.0 : g_huberScale / absoluteResidual;
                // scale H and b contributions by robustWeight; sum e*e UNWEIGHTED into s_acc[28]
                ...
                atomicAdd(s_acc[k++], int(round(robustWeight * J[r]*J[col] * SCALE)));
                ...
                atomicAdd(s_acc[21+r], int(round(robustWeight * (-J[r]*e) * SCALE)));
                atomicAdd(s_acc[27], 1);
                atomicAdd(s_acc[28], int(round(e*e*SCALE)));
            }
```
Keep the exact same weight/rejection math as the CPU so `SolveMatchesCpuOnCorner` holds. `IcpPC` in `dispatchCentred` sets `pc.huberScale`, `pc.normalCompatibilityCosine` from `params`.

- [ ] **Step 5: `Tracker.cpp` passes source normals** — both trackers now call `Solve(frame.pts, frame.nrm, target, prior, params)` / `AlignPointToPlaneIcp(frame.pts, frame.nrm, target, prior, params)`, and set `params.huberScale = model->voxel; params.maxCorrDist = <tighter, per Task 5 default>`.

- [ ] **Step 6: Build + harness (record) + suite green** (`SolveMatchesCpuOnCorner` must stay within tolerance — GPU/CPU consistency).

- [ ] **Step 7: Commit** — `feat(icp): Tier 2 robust correspondences (Huber + normal rejection, GPU+CPU)`.

---

## Task 5 — Tier 3: coarse-to-fine annealing + constant-velocity motion model

**Files:** Modify `icp_iterate.comp.glsl`, `GpuPointToPlaneIcp.cpp`, `PointToPlaneIcp.h`, `RegistrationThread.{h,cpp}`; Test `test/test_gpuIcp.cpp` + a `RegistrationThread`-level motion check.

- [ ] **Step 1: Failing tests** — (a) harness with a larger initial perturbation where coarse-to-fine must converge where fixed-gate did not (`EXPECT_LT(...)`); (b) a motion-model unit check: a straight-line pose sequence where the constant-velocity prior yields lower first-iteration error than the previous-pose prior.

- [ ] **Step 2: Annealing WITHOUT breaking the grid hoist**

Build the grid ONCE at the **coarsest** cell (the widest annealed distance): in `Solve`/`prepareCentred`, pass `params.maxCorrDist` (the widest) as the grid cell (unchanged — it already is). Then vary only the **distance filter** per iteration:
- Add `currentMaxCorrespondenceDistance` to the GPU push constant (reuse `g_maxCorr` — set it per iteration in `dispatchCentred`) and to the CPU `Nearest(p, currentMaxCorrespondenceDistance)` call. The grid neighbor scan (cell size = widest distance) is unchanged; only `bestD2 = current*current` tightens.
- In both `Solve` loops, compute a geometric schedule: `currentMaxCorrespondenceDistance = wide * pow(narrow/wide, iter/(maxIters-1))` from `wide = params.maxCorrDist` down to `narrow = 0.5 * model-voxel-scale` (pass `narrow` via a new `RegistrationParam` field `minCorrespondenceDistance`, default `0`, meaning "no annealing = use maxCorrDist"). Also anneal `huberScale` proportionally.
- `dispatchCentred` must accept the per-iteration distance (add a parameter `float currentMaxCorrespondenceDistance` and set `pc.maxCorr` from it instead of `m_pMaxCorr`). This keeps buffers/grid bound once (hoist intact) — only the push constant changes per iteration.

- [ ] **Step 3: Constant-velocity motion model in `RegistrationThread`**

In `RegistrationThread.cpp::Run`, track two poses back:
```cpp
        Eigen::Isometry3f previousPose = Eigen::Isometry3f::Identity();
        Eigen::Isometry3f previousPreviousPose = Eigen::Isometry3f::Identity();
        bool haveTwoPoses = false;
        ...
            const Eigen::Isometry3f prior =
                    haveTwoPoses ? previousPose * (previousPreviousPose.inverse() * previousPose)
                                 : previousPose;
            TrackingResult a = m_tracker->Track(frame, model.get(), prior);
            const Eigen::Isometry3f pose = a.valid ? a.pose : previousPose;
            if (a.valid) { previousPreviousPose = previousPose; previousPose = a.pose; haveTwoPoses = true; }
            else { haveTwoPoses = false; } // stale velocity after a dropped track
```
(Replace the current `prev`-only logic; keep the `TrackedFrame` push identical.)

- [ ] **Step 4: Build + both new tests + harness (record) + full suite green** (three targets; `SolveMatchesCpuOnCorner` within tolerance; `Pipeline.*` green — the motion-model change must not regress the pipeline tests).

- [ ] **Step 5: Commit** — `feat(icp): Tier 3 coarse-to-fine annealing + constant-velocity motion model`.

---

## Results note (fill during execution)

Record the harness numbers after each tier so the improvement is auditable:

```
              transErr   rotErr   reconNnRmse   residualRmse
baseline (T2)   ...        ...        ...           ...
+Tier1 (T3)     ...        ...        ...           ...
+Tier2 (T4)     ...        ...        ...           ...
+Tier3 (T5)     ...        ...        ...           ...
```

## Follow-ups (out of scope)

- Global↔local composition (relocalization fallback when tracking is lost) — the four trackers remain independent selectable strategies.
- Direct continuous-TSDF tracking (align to the trilinear-interpolated field instead of entry points) — the higher ceiling, larger effort.
- Full multi-resolution model pyramid (this plan anneals the distance filter instead).
