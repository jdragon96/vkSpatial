# Native Registration M1 (FPFH + matching + RANSAC + Ceres) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native CPU point-cloud **global registration** front-end (`Engine::Registration`) — voxel downsample → FPFH → feature matching → RANSAC coarse → Ceres robust refine → `{valid, T, inliers}` — that aligns raw scan frames for `DirectionalTSDF`, validated by known-transform recovery.

**Architecture:** New `EngineRegistration` static lib (pure Eigen + Ceres, no Vulkan — mirrors the existing `EngineBackend`), so the GPU core is untouched. Correspondences are reconstructed natively (KISS-Matcher used only as an algorithm reference, not linked); the robust SE(3) optimize is delegated to Ceres Solver.

**Tech Stack:** C++17, Eigen, **Ceres Solver** (already installed, `find_package(Ceres)`), GoogleTest. macOS + cross-platform. No TBB/OpenMP/Vulkan in this module.

## Global Constraints

- Namespace **`Engine::Registration`**, module dir **`src/Engine/Registration/`**, target **`EngineRegistration`** (own static lib; do NOT add these sources to `EngineSpatial` — the GPU core must stay Ceres-free, per spec "코어 빌드 불변").
- Dependency policy: **Ceres (+Eigen) allowed for the optimize step**; do NOT link TBB/ROBIN/TEASER++/KISS-Matcher. Non-optimize stages (downsample/FPFH/matching/RANSAC) are pure Eigen.
- Reference-only source (read to mirror the algorithm, never `#include`/link): KISS-Matcher `cpp/kiss_matcher/core/kiss_matcher/{FasterPFH,ROBINMatching,GncSolver}.*` at `/private/tmp/claude-501/-Users-sjy-Desktop-VulkanProject-VkLBVH/93734787-6ad3-4cd0-b9f0-8c0ad9cd5ba0/scratchpad/kiss-matcher/`.
- Use **provided normals** when present (scan frames carry `nx,ny,nz`); FPFH does NOT re-estimate normals in M1.
- **Validation oracle = known-transform recovery** (no external reference): perturb a cloud by a known `T_gt`, register, assert recovery within tolerance. This is the acceptance test for the end-to-end pipeline (Tasks 5/6) and the FPFH/matching correctness signal.
- Build: `cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests`. `example2` is broken at HEAD (stub `example2/ShadowMap.cpp` = `int main(){return 0;}` to configure — see memory `project-repo-build-worktree-gotchas`); build only the needed targets, never the whole project. `VULKAN_SDK=/usr/local`. Sources are GLOB'd → re-run `cmake -S . -B build` after adding a file. Device: Apple M4 Max.
- Commit after every task with a passing build + tests.

## File Structure

**New files:**
- `src/Engine/Registration/RegistrationTypes.h` — `RegistrationConfig`, `RegistrationResult`, `Fpfh33` alias, `Correspondence`.
- `src/Engine/Registration/Downsample.h` / `.cpp` — voxel-grid downsample (points + normals).
- `src/Engine/Registration/Fpfh.h` / `.cpp` — FPFH descriptor (radius-neighbour SPFH → weighted FPFH).
- `src/Engine/Registration/FeatureMatching.h` / `.cpp` — descriptor-space KNN + ratio/mutual test.
- `src/Engine/Registration/GlobalRegistration.h` / `.cpp` — `Estimate()` orchestrator: RANSAC coarse + Ceres refine.
- `test/test_registration.cpp` — stage unit tests + end-to-end known-transform recovery.
- `example2/registration_chair_demo.cpp` — register raw `scanData/frame_*.ply` → `DirectionalTSDF`.

**Modified files:**
- `CMakeLists.txt` (top) — `find_package(Ceres REQUIRED)`.
- `src/Engine/CMakeLists.txt` — add `EngineRegistration` static lib (links `Eigen3::Eigen`, `Ceres::ceres`).
- `test/CMakeLists.txt` — link `Engine::Registration` into `vkspatial_tests`.
- `example2/CMakeLists.txt` — add the chair demo (links `Engine::Spatial` + `Engine::Registration`).

---

## Task 1: Module scaffold + types + voxel downsample + Ceres wiring

**Files:**
- Create: `src/Engine/Registration/RegistrationTypes.h`, `Downsample.h`, `Downsample.cpp`
- Modify: `CMakeLists.txt`, `src/Engine/CMakeLists.txt`, `test/CMakeLists.txt`
- Test: `test/test_registration.cpp`

**Interfaces:**
- Produces:
  - `struct Engine::Registration::RegistrationConfig { float voxelSize = 5.0f; float normalRadiusGain = 3.0f; float fpfhRadiusGain = 5.0f; int numMaxCorr = 5000; float ransacInlierGain = 2.0f; int ransacIters = 5000; float ceresLossGain = 1.0f; };` — **plain constant defaults only** (C++ default member initializers cannot reference another member, so store *gains* and derive the physical values in the pipeline: `normalRadius = normalRadiusGain*voxelSize`, `fpfhRadius = fpfhRadiusGain*voxelSize`, `ransacInlierThr = ransacInlierGain*voxelSize`, `ceresLossScale = ceresLossGain*voxelSize`). mm-scale; caller overrides `voxelSize` per data.
  - `struct Engine::Registration::RegistrationResult { bool valid = false; Eigen::Matrix4f T = Eigen::Matrix4f::Identity(); size_t numInliers = 0; float fitness = 0.0f; };`
  - `using Fpfh33 = Eigen::Matrix<float, 33, 1>;`
  - `struct Correspondence { int srcIdx, tgtIdx; };`
  - `struct PointCloud { std::vector<Eigen::Vector3f> points, normals; };` (normals empty ⇒ absent)
  - `PointCloud Engine::Registration::DownsampleVoxel(const PointCloud& in, float voxelSize);` (centroid per occupied cell; averaged+renormalized normal if present)

- [ ] **Step 1: Wire Ceres into CMake, add the EngineRegistration lib skeleton**

`CMakeLists.txt` (top): after `find_package(Eigen3 REQUIRED)` add:
```cmake
find_package(Ceres REQUIRED)
```
`src/Engine/CMakeLists.txt`: after the `EngineBackend` block (mirror it), add:
```cmake
file(GLOB_RECURSE ENGINE_REGISTRATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Registration/*.cpp")
add_library(EngineRegistration STATIC ${ENGINE_REGISTRATION_SOURCES})
add_library(Engine::Registration ALIAS EngineRegistration)
target_link_libraries(EngineRegistration PUBLIC Eigen3::Eigen Ceres::ceres)
target_include_directories(EngineRegistration
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3)
```
`test/CMakeLists.txt`: add `Engine::Registration` to `vkspatial_tests`'s `target_link_libraries`.

- [ ] **Step 2: Write the failing test** — downsample reduces count and preserves the shape's extent

`test/test_registration.cpp`:
```cpp
#include "Engine/Registration/Downsample.h"
#include "Engine/Registration/RegistrationTypes.h"
#include <gtest/gtest.h>
using namespace Engine::Registration;

TEST(Registration, VoxelDownsampleReducesAndKeepsExtent) {
    PointCloud in;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j) {
            in.points.emplace_back(i * 0.25f, j * 0.25f, 0.0f); // dense 10x10mm plane, 0.25mm spacing
            in.normals.emplace_back(0, 0, 1);
        }
    PointCloud out = DownsampleVoxel(in, 1.0f); // 1mm cells → ~10x10 = ~100 pts
    EXPECT_LT(out.points.size(), in.points.size());
    EXPECT_GT(out.points.size(), 50u);
    EXPECT_EQ(out.normals.size(), out.points.size());
    // normals preserved (all +Z)
    for (auto& n : out.normals) EXPECT_NEAR(n.z(), 1.0f, 1e-3f);
}
```

- [ ] **Step 3: Run to verify it fails** — `cmake -S . -B build && cmake --build build --parallel --target vkspatial_tests` → FAIL (headers missing).

- [ ] **Step 4: Write `RegistrationTypes.h` + `Downsample.{h,cpp}`**

`Downsample.cpp` core (hash grid): map `Eigen::Vector3i(floor(p/voxelSize))` → accumulate sum+count (+normal sum); emit centroid per cell; normalize summed normal if `!in.normals.empty()`. Reuse the `Vector3i` hash style from `DirectionalTSDFTypes.h`/existing code.

- [ ] **Step 5: Build + run** — `IntegrationQuality`-style: `./build/test/vkspatial_tests --gtest_filter='Registration.VoxelDownsampleReducesAndKeepsExtent'` PASS; full suite still green.

- [ ] **Step 6: Commit** — `git add src/Engine/Registration/RegistrationTypes.h src/Engine/Registration/Downsample.h src/Engine/Registration/Downsample.cpp CMakeLists.txt src/Engine/CMakeLists.txt test/CMakeLists.txt test/test_registration.cpp && git commit -m "feat(registration): EngineRegistration lib + Ceres wiring + voxel downsample"`

---

## Task 2: FPFH descriptor

**Files:**
- Create: `src/Engine/Registration/Fpfh.h`, `Fpfh.cpp`
- Test: `test/test_registration.cpp` (append)

**Interfaces:**
- Consumes: `PointCloud`, `Fpfh33` (Task 1).
- Produces: `std::vector<Fpfh33> Engine::Registration::ComputeFpfh(const PointCloud& cloud, float normalRadius, float fpfhRadius);` (cloud MUST have normals)

**Algorithm (standard FPFH, Rusu 2009 — mirror KISS-Matcher `FasterPFH.cpp` but plain/CPU):**
- Radius neighbours via a CPU grid (reuse the Task-1 hash grid at cell=`fpfhRadius`, scan the 3×3×3 cell block, keep within radius) — do NOT use `Engine::Spatial` BVH (large-N bug).
- **SPFH(p)**: for each neighbour q within `normalRadius`, with source normal `n_p`, target `n_q`, `d = q - p`, `|d|>0`: Darboux frame `u = n_p`, `v = (d/|d|) × u` normalized, `w = u × v`; features `f1 = v·n_q`, `f2 = u·(d/|d|)`, `f3 = atan2(w·n_q, u·n_q)`. Bin each of f1∈[-1,1], f2∈[-1,1], f3∈[-π,π] into **11 bins**, accumulate 3 histograms (normalized to sum 1 each) → SPFH is 33-D (concatenation).
- **FPFH(p)** `= SPFH(p) + (1/k) Σ_{q∈N(p, fpfhRadius)} (1/|q-p|) · SPFH(q)`; renormalize.

- [ ] **Step 1: Write the failing test** — FPFH is (approximately) rotation-invariant

```cpp
#include "Engine/Registration/Fpfh.h"
#include <Eigen/Geometry>

static Engine::Registration::PointCloud makeSphere(int n, float r) {
    Engine::Registration::PointCloud c;
    for (int i = 0; i < n; ++i) {
        float a = 2.399963f * i, z = 1.0f - 2.0f * (i + 0.5f) / n;
        float rr = std::sqrt(std::max(0.0f, 1 - z*z));
        Eigen::Vector3f d(rr*std::cos(a), rr*std::sin(a), z);
        c.points.push_back(r * d); c.normals.push_back(d); // outward normals
    }
    return c;
}

TEST(Registration, FpfhIsApproximatelyRotationInvariant) {
    auto s = makeSphere(600, 20.0f);
    Eigen::Matrix3f R = Eigen::AngleAxisf(0.7f, Eigen::Vector3f(0.3f,0.8f,0.5f).normalized()).toRotationMatrix();
    Engine::Registration::PointCloud sr = s;
    for (auto& p : sr.points) p = R * p;
    for (auto& nrm : sr.normals) nrm = R * nrm;
    auto f0 = Engine::Registration::ComputeFpfh(s,  60.0f, 100.0f);
    auto f1 = Engine::Registration::ComputeFpfh(sr, 60.0f, 100.0f);
    // point i maps to point i under R (same ordering), so descriptors should be close
    double maxdiff = 0;
    for (size_t i = 0; i < f0.size(); ++i) maxdiff = std::max<double>(maxdiff, (f0[i]-f1[i]).norm());
    EXPECT_LT(maxdiff, 5.0) << "FPFH not rotation-invariant enough (max L2 " << maxdiff << ")";
}
```
(Tune the tolerance once measured; the point is it must be SMALL relative to descriptor magnitude — if it's near the descriptor norm, FPFH is wrong. Keep `n` under ~1000.)

- [ ] **Step 2: Run to verify it fails** — link error (`ComputeFpfh` undefined).

- [ ] **Step 3: Implement `Fpfh.{h,cpp}`** per the algorithm above. Cross-check the three features and the weighted-sum formula against KISS-Matcher `FasterPFH.cpp` `ComputePointSPFHSignature`/`WeightPointSPFHSignature` (reference only).

- [ ] **Step 4: Build + run** — `--gtest_filter='Registration.Fpfh*'` PASS; full suite green.

- [ ] **Step 5: Commit** — `git add src/Engine/Registration/Fpfh.h src/Engine/Registration/Fpfh.cpp test/test_registration.cpp && git commit -m "feat(registration): FPFH descriptor (rotation-invariant)"`

---

## Task 3: Feature matching

**Files:**
- Create: `src/Engine/Registration/FeatureMatching.h`, `FeatureMatching.cpp`
- Test: `test/test_registration.cpp` (append)

**Interfaces:**
- Consumes: `Fpfh33`, `Correspondence` (Task 1/2).
- Produces: `std::vector<Correspondence> Engine::Registration::MatchFeatures(const std::vector<Fpfh33>& srcF, const std::vector<Fpfh33>& tgtF, float ratioThr = 0.95f, int numMaxCorr = 5000);` (src→tgt nearest in 33-D by L2, keep if 1st/2nd-NN ratio < `ratioThr`; cap at `numMaxCorr` by best ratio)

- [ ] **Step 1: Write the failing test** — a rotated copy matches each point to itself

```cpp
#include "Engine/Registration/FeatureMatching.h"
TEST(Registration, MatchRecoversIdentityCorrespondencesUnderRotation) {
    auto s = makeSphere(500, 20.0f);
    Eigen::Matrix3f R = Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    Engine::Registration::PointCloud sr = s;
    for (auto& p : sr.points) p = R * p;
    for (auto& n : sr.normals) n = R * n;
    auto fs = Engine::Registration::ComputeFpfh(s,  60.0f, 100.0f);
    auto ft = Engine::Registration::ComputeFpfh(sr, 60.0f, 100.0f);
    auto corr = Engine::Registration::MatchFeatures(fs, ft);
    // most correspondences should be i→i (descriptor space is symmetric under R)
    int selfMatches = 0; for (auto& c : corr) if (c.srcIdx == c.tgtIdx) ++selfMatches;
    EXPECT_GT(corr.size(), 100u);
    EXPECT_GT(double(selfMatches) / corr.size(), 0.5) << selfMatches << "/" << corr.size();
}
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `FeatureMatching.{h,cpp}`** — brute-force or a small kd-tree over 33-D `tgtF` (N is downsampled/keypoint scale, so O(N²) brute force is fine for M1; note it as an M2 speedup).
- [ ] **Step 4: Build + run** — `--gtest_filter='Registration.Match*'` PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(registration): FPFH feature matching (ratio test)"`

---

## Task 4: RANSAC coarse solver + end-to-end known-transform recovery

**Files:**
- Create: `src/Engine/Registration/GlobalRegistration.h`, `GlobalRegistration.cpp`
- Test: `test/test_registration.cpp` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `Eigen::Matrix4f Engine::Registration::SolveRigidUmeyama(const std::vector<Eigen::Vector3f>& src, const std::vector<Eigen::Vector3f>& dst);` (closed-form, no scale)
  - `RegistrationResult Engine::Registration::EstimateRansac(const PointCloud& src, const PointCloud& tgt, const RegistrationConfig& cfg);` (full pipeline: downsample → FPFH → match → RANSAC; NO Ceres yet)

**RANSAC:** repeat `cfg.ransacIters`: sample 3 correspondences, `SolveRigidUmeyama` on those, count inliers (`‖T·p_src − p_tgt‖ < inlierThr` where `inlierThr = cfg.ransacInlierGain*cfg.voxelSize`, over ALL correspondences); keep max-inlier T; final `SolveRigidUmeyama` on all inliers. `valid = numInliers >= 3 && fitness > minFitness`.

- [ ] **Step 1: Write the failing test** — **the acceptance test**: recover a known transform

```cpp
#include "Engine/Registration/GlobalRegistration.h"
TEST(Registration, RansacRecoversKnownTransform) {
    auto tgt = makeSphere(600, 20.0f);           // "model"
    // bumpy sphere so FPFH isn't degenerate: perturb radius by a hash
    for (size_t i = 0; i < tgt.points.size(); ++i) tgt.points[i] *= (1.0f + 0.15f*std::sin(0.7f*i));
    Eigen::Matrix3f Rgt = Eigen::AngleAxisf(0.6f, Eigen::Vector3f(0.2f,0.7f,0.6f).normalized()).toRotationMatrix();
    Eigen::Vector3f tgt_t(8.0f, -5.0f, 3.0f);
    Engine::Registration::PointCloud src = tgt;   // src = model moved by Tgt
    for (size_t i = 0; i < src.points.size(); ++i) { src.points[i] = Rgt*tgt.points[i] + tgt_t; src.normals[i] = Rgt*tgt.normals[i]; }
    Engine::Registration::RegistrationConfig cfg; cfg.voxelSize = 2.0f;
    auto res = Engine::Registration::EstimateRansac(src, tgt, cfg);  // aligns src→tgt ⇒ T ≈ [Rgt|tgt_t]^-1
    ASSERT_TRUE(res.valid);
    Eigen::Matrix4f Tgt = Eigen::Matrix4f::Identity(); Tgt.block<3,3>(0,0)=Rgt; Tgt.block<3,1>(0,3)=tgt_t;
    Eigen::Matrix4f err = res.T * Tgt;   // should be ≈ identity
    float rotErr = Eigen::AngleAxisf(Eigen::Matrix3f(err.block<3,3>(0,0))).angle();
    float trErr  = err.block<3,1>(0,3).norm();
    EXPECT_LT(rotErr, 0.1f) << "rot err rad";       // ~6°
    EXPECT_LT(trErr, 3.0f)  << "trans err mm";      // coarse RANSAC tolerance
}
```
(If recovery is unstable, first assert on a rotation-only or smaller transform, then widen — but the test MUST prove real recovery, not a trivial pass. Keep point counts < 1000.)

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `GlobalRegistration.{h,cpp}`** (`SolveRigidUmeyama` via Eigen `umeyama` with `with_scaling=false`, `EstimateRansac`).
- [ ] **Step 4: Build + run** — the recovery test PASSES; full suite green. This is the first end-to-end registration.
- [ ] **Step 5: Commit** — `git commit -m "feat(registration): RANSAC coarse global registration (known-transform recovery)"`

---

## Task 5: Ceres robust refinement

**Files:**
- Modify: `src/Engine/Registration/GlobalRegistration.h`, `GlobalRegistration.cpp`
- Test: `test/test_registration.cpp` (append)

**Interfaces:**
- Produces: `RegistrationResult Engine::Registration::Estimate(const PointCloud& src, const PointCloud& tgt, const RegistrationConfig& cfg);` — runs `EstimateRansac` for the coarse T, then a Ceres robust refine over the inlier correspondences, returning the refined result.

**Ceres problem:** parameters `double q[4]` (`ceres::QuaternionManifold`) + `double t[3]`, seeded from the coarse T. One residual block per inlier correspondence: `r = R(q)·p_src_i + t − p_tgt_i` (3-vector), wrapped in `ceres::CauchyLoss(cfg.ceresLossGain*cfg.voxelSize)`. Use `ceres::AutoDiffCostFunction`. Solve with `ceres::Solver` (dense QR, silent). Compose refined `T`; recount inliers/fitness.

- [ ] **Step 1: Write the failing test** — Ceres refine tightens recovery accuracy vs RANSAC-only

```cpp
TEST(Registration, CeresRefineTightensRecovery) {
    // same fixture as RansacRecoversKnownTransform (factor it into a helper)
    // ... build src/tgt with known Tgt ...
    auto coarse = Engine::Registration::EstimateRansac(src, tgt, cfg);
    auto refined = Engine::Registration::Estimate(src, tgt, cfg);
    ASSERT_TRUE(refined.valid);
    Eigen::Matrix4f errC = coarse.T * Tgt, errR = refined.T * Tgt;
    float rotC = Eigen::AngleAxisf(Eigen::Matrix3f(errC.block<3,3>(0,0))).angle();
    float rotR = Eigen::AngleAxisf(Eigen::Matrix3f(errR.block<3,3>(0,0))).angle();
    EXPECT_LT(rotR, 0.03f) << "refined rot err rad (~1.7°)";
    EXPECT_LE(rotR, rotC + 1e-4f) << "Ceres refine should not worsen the coarse estimate";
}
```

- [ ] **Step 2: Run to verify it fails** (`Estimate` undefined).
- [ ] **Step 3: Implement `Estimate` with the Ceres refine** (add a `RigidResidual` functor). Guard: if coarse `!valid`, return coarse (don't refine garbage).
- [ ] **Step 4: Build + run** — refine test PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -m "feat(registration): Ceres robust SE(3) refinement (Cauchy loss)"`

---

## Task 6: Chair registration demo (raw frames → DirectionalTSDF)

**Files:**
- Create: `example2/registration_chair_demo.cpp`
- Modify: `example2/CMakeLists.txt`
- (No unit test — this is a demonstration executable requiring external scan data.)

**Interfaces:**
- Consumes: `Engine::Registration::Estimate`, `Engine::Spatial::DirectionalTSDF`.

**Demo:** load `scanData/frame_*.ply` (reuse the ASCII PLY parse from `directional_tsdf_chair_benchmark.cpp`), then reconstruct WITHOUT assuming pre-registration: keep an accumulating pose `T_world`; for frame f>0 register frame f (src) to frame f-1 (tgt) via `Estimate`, compose `T_world`; transform points+normals by `T_world`; `DirectionalTSDF.Integrate` (Unified backend). Print per-frame inliers/fitness/rot+trans delta; export `chair_registered_recon.ply`. Gate integration on `res.valid`.

- [ ] **Step 1: Write the demo** (`add_spatial_example(registration_chair_demo registration_chair_demo.cpp)` + link `Engine::Registration`; the `add_spatial_example` helper already links `Engine::Spatial`).
- [ ] **Step 2: Build** — `cmake -S . -B build && cmake --build build --parallel --target registration_chair_demo`.
- [ ] **Step 3: Run on real data** — `./build/example2/registration_chair_demo /Users/sjy/Desktop/VulkanProject/VkLBVH/scanData 8`. Expected: consecutive-frame registrations report small rot/trans deltas (frames are ~registered already) and high inlier counts; a non-empty `chair_registered_recon.ply`. Since the frames are already in a common frame, the recovered per-frame relative transforms should be near-identity — that is the sanity signal.
- [ ] **Step 4: Commit** — `git add example2/registration_chair_demo.cpp example2/CMakeLists.txt && git commit -m "feat(example2): raw-frame registration demo feeding DirectionalTSDF"`

---

## Out of scope (follow-on)
- **M2**: GPU (Vulkan compute) FPFH + matching; needs the large-N BVH bug fixed or aggressive downsampling. kd-tree for matching (replace O(N²)).
- **M3**: Ceres-GNC (μ schedule) or vendored TEASER GNC for extreme-outlier robustness.
- **Local ICP** (fine, frame-to-model with motion prior) and **loop closure / pose-graph** (`EngineBackend`).
- Windows CI verification.

## Self-Review Notes
- **Spec coverage:** downsample → Task 1; FPFH → Task 2; matching → Task 3; RANSAC coarse → Task 4; Ceres refine → Task 5; TSDF demo + known-transform recovery validation → Tasks 4/5/6. M2/M3 deferred per spec.
- **Dependency policy honored:** Ceres only in `EngineRegistration`; `EngineSpatial` (GPU core) untouched; no TBB/ROBIN/TEASER++ link.
- **Discriminating tests:** each stage test would fail if that stage were wrong (rotation-invariance for FPFH, self-match ratio for matching, known-transform recovery for RANSAC/Ceres) — not trivial passes.
- **Pinned values:** voxelSize (data-scale, mm), 11 bins/feature (33-D FPFH), ratioThr 0.95, CauchyLoss scale = voxelSize, recovery tolerances (rot <0.1 coarse / <0.03 refined rad). All tunable; keep test point counts < 1000 (Engine::Core large-N caveat, though this module is CPU-only so it is not directly affected).
