# DirectionalTSDF Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synthetic-dataset + ground-truth + RMSE harness that quantifies how accurately `Engine::Spatial::DirectionalTSDF`'s integrate→extract pipeline reconstructs analytic surfaces (sphere, plane), reported as an accuracy RMSE (point-to-surface) and a completeness RMSE (GT→recon nearest neighbour).

**Architecture:** Three header-only CPU utilities under `src/Engine/Eval/` (`SyntheticSurface.h`, `ScanSampler.h`, `RmseMetrics.h`), an `example2` driver that scans → reconstructs → prints RMSE + exports PLYs, and — after the real numbers are observed — a GTest regression guard plus pure-CPU metric unit tests. Header-only means no new CMake library target; both `example2` and `test` already have `${CMAKE_SOURCE_DIR}/src` on their include path.

**Tech Stack:** C++17, Eigen, `Engine::Core::Context`, `Engine::Spatial::DirectionalTSDF`, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-eval-design.md` — read it if any step here is ambiguous.

## Global Constraints

- All new code lives in `namespace Engine::Eval`. Header-only utilities: every function/method `inline` (or a class member defined in-class) so multiple translation units can include them without ODR violations.
- Only **positions** of reconstructed points are used for RMSE (normals excluded — spec §범위 밖). `DirectionalTSDF::PointCloud()` returns `const std::vector<ExtractedPoint>&`; use `pt.position` (an `Eigen::Vector3f`).
- Empty reconstruction must yield `std::numeric_limits<float>::infinity()` from the RMSE functions so a regression guard necessarily fails on a broken reconstruction.
- `DirectionalTSDF`/`Engine::Spatial`/`Engine::Compute`/`Engine::Core` source is NOT modified by this plan.
- Do not modify `src/Engine/CMakeLists.txt` (header-only, no target). `example2/CMakeLists.txt` gets one new executable block; `test/CMakeLists.txt` needs no change (its `file(GLOB ...)` picks up new `test_*.cpp`).
- Build/run commands (VULKAN_SDK must be exported):
  ```bash
  export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
  cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
  cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
  cmake --build build --target directional_tsdf_eval -j"$(sysctl -n hw.ncpu)"
  ```
- Test baseline before this plan: full suite passes except the known pre-existing `WideBVHTest.RadiusMatchesCpuReference`.

---

### Task 1: `RmseMetrics.h` + pure-CPU unit tests

This task is fully CPU and needs no GPU/DirectionalTSDF, so it is the cleanest place to start and its unit tests are the fastest in the suite.

**Files:**
- Create: `src/Engine/Eval/RmseMetrics.h`
- Create: `src/Engine/Eval/SyntheticSurface.h` (only the abstract `Surface` base + `PlaneSurface`, needed by the metric tests; `SphereSurface` is added in Task 2)
- Create: `test/test_directionalTSDFEval.cpp`

**Interfaces:**
- Produces:
  ```cpp
  namespace Engine::Eval {
      class Surface { // abstract
          virtual float Distance(const Eigen::Vector3f &p) const = 0;
          virtual Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const = 0;
          virtual std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const = 0;
      };
      class PlaneSurface : public Surface { PlaneSurface(point, normal, uExtent, vExtent); ... };
      float AccuracyRMSE(const std::vector<Eigen::Vector3f>&, const Surface&);
      float CompletenessRMSE(const std::vector<Eigen::Vector3f>& gtDense, const std::vector<Eigen::Vector3f>& recon);
      float ReconToGtNnRMSE(const std::vector<Eigen::Vector3f>& recon, const std::vector<Eigen::Vector3f>& gtDense);
  }
  ```

- [ ] **Step 1: Write `SyntheticSurface.h` with the base + `PlaneSurface`**

Create `src/Engine/Eval/SyntheticSurface.h`:

```cpp
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Engine::Eval {

    // Analytic surface with a closed-form unsigned distance, a surface normal, and a dense
    // sampler for ground-truth point clouds. Header-only (all methods inline).
    class Surface {
    public:
        virtual ~Surface() = default;
        virtual float Distance(const Eigen::Vector3f &p) const = 0;
        virtual Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const = 0;
        virtual std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const = 0;
    };

    // Finite plane patch: point q, unit normal n, in-plane half-extents (uExtent, vExtent).
    // Distance is the infinite-plane perpendicular distance (recon points only appear near
    // the scanned patch, so the finite extent does not affect the metric in practice).
    class PlaneSurface : public Surface {
    public:
        PlaneSurface(const Eigen::Vector3f &point, const Eigen::Vector3f &normal,
                     float uExtent, float vExtent)
            : m_point(point), m_normal(normal.normalized()),
              m_uExtent(uExtent), m_vExtent(vExtent) {
            // Build an orthonormal in-plane basis (m_u, m_v) ⟂ m_normal.
            Eigen::Vector3f seed = std::abs(m_normal.x()) < 0.9f
                                           ? Eigen::Vector3f(1, 0, 0)
                                           : Eigen::Vector3f(0, 1, 0);
            m_u = (seed - m_normal * seed.dot(m_normal)).normalized();
            m_v = m_normal.cross(m_u);
        }

        float Distance(const Eigen::Vector3f &p) const override {
            return std::abs((p - m_point).dot(m_normal));
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &) const override { return m_normal; }

        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const int n = std::max(2, int(std::sqrt(double(approxCount))));
            std::vector<Eigen::Vector3f> out;
            out.reserve(size_t(n) * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const float u = -m_uExtent + 2.0f * m_uExtent * float(i) / float(n - 1);
                    const float v = -m_vExtent + 2.0f * m_vExtent * float(j) / float(n - 1);
                    out.push_back(m_point + m_u * u + m_v * v);
                }
            return out;
        }

    private:
        Eigen::Vector3f m_point, m_normal, m_u, m_v;
        float m_uExtent, m_vExtent;
    };

} // namespace Engine::Eval
```

- [ ] **Step 2: Write `RmseMetrics.h`**

Create `src/Engine/Eval/RmseMetrics.h`:

```cpp
#pragma once

#include "Engine/Eval/SyntheticSurface.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <vector>

namespace Engine::Eval {

    // Accuracy: how close reconstructed points sit to the true surface (exact, closed-form).
    // Empty reconstruction → +inf so a regression guard necessarily fails.
    inline float AccuracyRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                              const Surface &surface) {
        if (reconPoints.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &p : reconPoints) {
            const float d = surface.Distance(p);
            sumSq += double(d) * double(d);
        }
        return float(std::sqrt(sumSq / double(reconPoints.size())));
    }

    // Nearest-neighbour distance RMSE from each point in `from` to the closest in `to`
    // (CPU brute-force). Empty `from` or `to` → +inf.
    inline float NearestNeighbourRMSE(const std::vector<Eigen::Vector3f> &from,
                                      const std::vector<Eigen::Vector3f> &to) {
        if (from.empty() || to.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &a : from) {
            float best = std::numeric_limits<float>::max();
            for (const auto &b : to) {
                const float d2 = (a - b).squaredNorm();
                if (d2 < best) best = d2;
            }
            sumSq += double(best);
        }
        return float(std::sqrt(sumSq / double(from.size())));
    }

    // Completeness: how well the true surface is covered by reconstruction —
    // for each GT point, nearest reconstructed point distance.
    inline float CompletenessRMSE(const std::vector<Eigen::Vector3f> &gtDense,
                                  const std::vector<Eigen::Vector3f> &reconPoints) {
        return NearestNeighbourRMSE(gtDense, reconPoints);
    }

    // Cross-check accuracy via nearest GT (bounded by GT sampling density).
    inline float ReconToGtNnRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                                 const std::vector<Eigen::Vector3f> &gtDense) {
        return NearestNeighbourRMSE(reconPoints, gtDense);
    }

} // namespace Engine::Eval
```

- [ ] **Step 3: Write the pure-CPU metric unit tests (no GPU)**

Create `test/test_directionalTSDFEval.cpp`:

```cpp
#include <gtest/gtest.h>

#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Eval/SyntheticSurface.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace Engine::Eval;

TEST(RmseMetricsTest, AccuracyToPlaneMatchesHandComputation) {
    // Plane z=0. Points at z = {0.1, -0.2, 0.0}. distances {0.1, 0.2, 0.0}.
    // RMSE = sqrt((0.01 + 0.04 + 0.0)/3) = sqrt(0.05/3) ≈ 0.129099.
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    std::vector<Eigen::Vector3f> recon = {
            {0.3f, -0.4f, 0.1f}, {-0.1f, 0.2f, -0.2f}, {0.5f, 0.5f, 0.0f}};
    EXPECT_NEAR(AccuracyRMSE(recon, plane), 0.1290994f, 1e-5f);
}

TEST(RmseMetricsTest, NearestNeighbourMatchesHandComputation) {
    // from = {(0,0,0)}, to = {(3,4,0),(1,0,0)}. nearest to (0,0,0) is (1,0,0) at dist 1.
    std::vector<Eigen::Vector3f> from = {{0, 0, 0}};
    std::vector<Eigen::Vector3f> to = {{3, 4, 0}, {1, 0, 0}};
    EXPECT_NEAR(NearestNeighbourRMSE(from, to), 1.0f, 1e-6f);
}

TEST(RmseMetricsTest, EmptyInputsYieldInfinity) {
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    std::vector<Eigen::Vector3f> empty;
    std::vector<Eigen::Vector3f> some = {{0, 0, 0}};
    EXPECT_TRUE(std::isinf(AccuracyRMSE(empty, plane)));
    EXPECT_TRUE(std::isinf(CompletenessRMSE(some, empty)));   // recon empty
    EXPECT_TRUE(std::isinf(CompletenessRMSE(empty, some)));   // gt empty
}

TEST(SyntheticSurfaceTest, PlaneDistanceAndDenseSampling) {
    PlaneSurface plane(Eigen::Vector3f(0, 0, 5), Eigen::Vector3f(0, 0, 1), 2.0f, 3.0f);
    EXPECT_NEAR(plane.Distance(Eigen::Vector3f(1, 2, 5.25f)), 0.25f, 1e-6f);
    auto dense = plane.SampleDense(400);
    ASSERT_GE(dense.size(), 100u);
    for (const auto &p : dense) {
        EXPECT_NEAR(p.z(), 5.0f, 1e-5f);       // on the plane
        EXPECT_LE(std::abs(p.x()), 2.0f + 1e-4f);
        EXPECT_LE(std::abs(p.y()), 3.0f + 1e-4f);
    }
}
```

- [ ] **Step 4: Build and run the CPU metric tests**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="RmseMetricsTest.*:SyntheticSurfaceTest.*"
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Eval/RmseMetrics.h src/Engine/Eval/SyntheticSurface.h test/test_directionalTSDFEval.cpp
git commit -m "Add Engine::Eval RMSE metrics + plane surface (header-only, CPU tests)"
```

---

### Task 2: `SphereSurface` + `ScanSampler.h`

**Files:**
- Modify: `src/Engine/Eval/SyntheticSurface.h` (add `SphereSurface`)
- Create: `src/Engine/Eval/ScanSampler.h`
- Modify: `test/test_directionalTSDFEval.cpp` (append CPU tests)

**Interfaces:**
- Consumes: `Surface` (Task 1).
- Produces:
  ```cpp
  namespace Engine::Eval {
      class SphereSurface : public Surface { SphereSurface(center, radius); ... };
      struct ScanFrame { std::vector<Eigen::Vector3f> points, normals; Eigen::Vector3f cameraPos, aabbCenterHint; };
      struct OrbitParams { std::vector<Eigen::Vector3f> surfaceSamples; float cameraRadius; Eigen::Vector3f orbitCenter; int numElevation, numAzimuth; float cosVisibility; };
      std::vector<ScanFrame> GenerateOrbitScan(const Surface&, const OrbitParams&);
  }
  ```

- [ ] **Step 1: Add `SphereSurface` to `SyntheticSurface.h`**

In `src/Engine/Eval/SyntheticSurface.h`, before the closing `} // namespace Engine::Eval`, add:

```cpp
    // Sphere centred at `center` with radius `radius`.
    class SphereSurface : public Surface {
    public:
        SphereSurface(const Eigen::Vector3f &center, float radius)
            : m_center(center), m_radius(radius) {}

        float Distance(const Eigen::Vector3f &p) const override {
            return std::abs((p - m_center).norm() - m_radius);
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f d = p - m_center;
            const float n = d.norm();
            return n > 1e-8f ? Eigen::Vector3f(d / n) : Eigen::Vector3f(0, 0, 1);
        }

        // Roughly uniform surface sampling via the Fibonacci sphere.
        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const uint32_t n = std::max<uint32_t>(4, approxCount);
            std::vector<Eigen::Vector3f> out;
            out.reserve(n);
            const float golden = float(M_PI) * (3.0f - std::sqrt(5.0f)); // golden angle
            for (uint32_t i = 0; i < n; ++i) {
                const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(n); // (-1, 1)
                const float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
                const float theta = golden * float(i);
                out.push_back(m_center + m_radius * Eigen::Vector3f(std::cos(theta) * r, y,
                                                                    std::sin(theta) * r));
            }
            return out;
        }

    private:
        Eigen::Vector3f m_center;
        float m_radius;
    };
```

- [ ] **Step 2: Write `ScanSampler.h`**

Create `src/Engine/Eval/ScanSampler.h`:

```cpp
#pragma once

#include "Engine/Eval/SyntheticSurface.h"

#include <Eigen/Core>
#include <cmath>
#include <vector>

namespace Engine::Eval {

    struct ScanFrame {
        std::vector<Eigen::Vector3f> points;
        std::vector<Eigen::Vector3f> normals;
        Eigen::Vector3f cameraPos = Eigen::Vector3f::Zero();
        Eigen::Vector3f aabbCenterHint = Eigen::Vector3f::Zero();
    };

    struct OrbitParams {
        std::vector<Eigen::Vector3f> surfaceSamples; // candidate surface points to scan
        float cameraRadius = 3.0f;
        Eigen::Vector3f orbitCenter = Eigen::Vector3f::Zero();
        int numElevation = 9;        // latitude rings, −80°..+80°
        int numAzimuth = 36;         // longitudes per ring
        float cosVisibility = 0.15f; // reject grazing samples
    };

    // One ScanFrame per camera pose. A surface sample is included in a frame if its outward
    // normal faces the camera (normal · dir(sample→cam) > cosVisibility). Mirrors the
    // visibility logic in example2/voxel_tsdf_mc.cpp so multi-view running-average and
    // directional layering are exercised. Empty frames are skipped.
    inline std::vector<ScanFrame> GenerateOrbitScan(const Surface &surface,
                                                    const OrbitParams &params) {
        std::vector<ScanFrame> frames;
        for (int el = 0; el < params.numElevation; ++el) {
            const float elDeg = params.numElevation > 1
                                        ? -80.0f + 160.0f * float(el) / float(params.numElevation - 1)
                                        : 0.0f;
            const float elRad = elDeg * float(M_PI) / 180.0f;
            const float cosEl = std::cos(elRad), sinEl = std::sin(elRad);

            for (int az = 0; az < params.numAzimuth; ++az) {
                const float azRad = 2.0f * float(M_PI) * float(az) / float(params.numAzimuth);
                const Eigen::Vector3f cam =
                        params.orbitCenter +
                        params.cameraRadius * Eigen::Vector3f(cosEl * std::cos(azRad), sinEl,
                                                              cosEl * std::sin(azRad));

                ScanFrame frame;
                frame.cameraPos = cam;
                Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                for (const auto &p : params.surfaceSamples) {
                    const Eigen::Vector3f toCam = (cam - p);
                    const float len = toCam.norm();
                    if (len < 1e-8f) continue;
                    const Eigen::Vector3f dir = toCam / len;
                    if (surface.NormalAt(p).dot(dir) > params.cosVisibility) {
                        frame.points.push_back(p);
                        frame.normals.push_back(surface.NormalAt(p));
                        centroid += p;
                    }
                }
                if (frame.points.empty()) continue;
                frame.aabbCenterHint = centroid / float(frame.points.size());
                frames.push_back(std::move(frame));
            }
        }
        return frames;
    }

} // namespace Engine::Eval
```

- [ ] **Step 3: Append CPU tests for the new pieces**

Append to `test/test_directionalTSDFEval.cpp` (add `#include "Engine/Eval/ScanSampler.h"` at the top):

```cpp
TEST(SyntheticSurfaceTest, SphereDistanceAndDenseSampling) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    EXPECT_NEAR(sphere.Distance(Eigen::Vector3f(2, 0, 0)), 1.0f, 1e-6f);   // outside
    EXPECT_NEAR(sphere.Distance(Eigen::Vector3f(0.5f, 0, 0)), 0.5f, 1e-6f); // inside
    auto dense = sphere.SampleDense(2000);
    ASSERT_EQ(dense.size(), 2000u);
    for (const auto &p : dense) EXPECT_NEAR(p.norm(), 1.0f, 1e-4f); // on the unit sphere
}

TEST(ScanSamplerTest, SphereScanCoversAllSamplesAcrossViews) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    OrbitParams params;
    params.surfaceSamples = sphere.SampleDense(1000);
    params.cameraRadius = 3.0f;

    auto frames = GenerateOrbitScan(sphere, params);
    ASSERT_FALSE(frames.empty());

    // Every frame's samples must face their camera, and the union over all frames must
    // cover (nearly) every surface sample — a sphere is fully visible across the orbit.
    std::vector<bool> seen(params.surfaceSamples.size(), false);
    for (const auto &f : frames) {
        ASSERT_EQ(f.points.size(), f.normals.size());
        for (size_t i = 0; i < params.surfaceSamples.size(); ++i)
            for (const auto &pt : f.points)
                if ((pt - params.surfaceSamples[i]).squaredNorm() < 1e-10f) seen[i] = true;
    }
    size_t covered = 0;
    for (bool b : seen) covered += b ? 1 : 0;
    EXPECT_GT(covered, params.surfaceSamples.size() * 95 / 100); // ≥95% covered
}
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="RmseMetricsTest.*:SyntheticSurfaceTest.*:ScanSamplerTest.*"
```
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/Eval/SyntheticSurface.h src/Engine/Eval/ScanSampler.h test/test_directionalTSDFEval.cpp
git commit -m "Add SphereSurface + orbit ScanSampler to Engine::Eval (CPU tests)"
```

---

### Task 3: `directional_tsdf_eval` example — reconstruct + report RMSE

**Files:**
- Create: `example2/directional_tsdf_eval.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: all of `Engine::Eval` (Tasks 1–2), `Engine::Spatial::DirectionalTSDF`, `Engine::Core::Context`.
- Produces: the `directional_tsdf_eval` executable printing accuracy/completeness/cross RMSE for sphere and plane, and exporting `recon_sphere.ply`/`gt_sphere.ply`/`recon_plane.ply`/`gt_plane.ply`.

- [ ] **Step 1: Write the example**

Create `example2/directional_tsdf_eval.cpp`:

```cpp
// Quantitative evaluation of DirectionalTSDF integrate→extract against analytic ground
// truth. For each analytic surface (sphere, plane): synthesise an orbit scan, reconstruct,
// then report accuracy RMSE (point-to-surface) and completeness RMSE (GT→recon NN).
#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Eval/ScanSampler.h"
#include "Engine/Eval/SyntheticSurface.h"
#include "Engine/Spatial/DirectionalTSDF.h"

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace Engine::Eval;

static void savePLY(const std::string &path, const std::vector<Eigen::Vector3f> &pts) {
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("savePLY: cannot open " + path);
    f << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
      << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto &p : pts) f << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
}

static std::vector<Eigen::Vector3f> reconPositions(const Engine::Spatial::DirectionalTSDF &tsdf) {
    std::vector<Eigen::Vector3f> out;
    out.reserve(tsdf.PointCloud().size());
    for (const auto &pt : tsdf.PointCloud()) out.push_back(pt.position);
    return out;
}

static void evalSurface(Engine::Core::Context &ctx, const std::string &name,
                        const Surface &surface, float cameraRadius, const Eigen::Vector3f &orbitCenter) {
    OrbitParams params;
    params.surfaceSamples = surface.SampleDense(4000); // scan candidates
    params.cameraRadius = cameraRadius;
    params.orbitCenter = orbitCenter;
    auto frames = GenerateOrbitScan(surface, params);

    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f);
    int totalPts = 0;
    for (const auto &fr : frames) {
        tsdf.Integrate(fr.points, fr.normals, fr.cameraPos, fr.aabbCenterHint);
        totalPts += int(fr.points.size());
    }

    const auto recon = reconPositions(tsdf);
    const auto gt = surface.SampleDense(8000); // independent dense GT
    const float acc = AccuracyRMSE(recon, surface);
    const float comp = CompletenessRMSE(gt, recon);
    const float cross = ReconToGtNnRMSE(recon, gt);

    std::cout << "=== " << name << " ===\n"
              << "  frames=" << frames.size() << " integrated_pts=" << totalPts
              << " recon_pts=" << recon.size() << " gt_pts=" << gt.size() << "\n"
              << "  accuracyRMSE (point-to-surface) = " << acc << "\n"
              << "  completenessRMSE (GT->recon NN)  = " << comp << "\n"
              << "  reconToGtNnRMSE (recon->GT NN)   = " << cross << "\n";

    savePLY("recon_" + name + ".ply", recon);
    savePLY("gt_" + name + ".ply", gt);
}

int main() {
    Engine::Core::Context ctx;
    // Sphere: radius 1 at origin, camera orbit at 3.0.
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    evalSurface(ctx, "sphere", sphere, 3.0f, Eigen::Vector3f(0, 0, 0));

    // Plane: z=0 patch, camera orbit centred above it so views look down at the patch.
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    evalSurface(ctx, "plane", plane, 3.0f, Eigen::Vector3f(0, 0, 1.5f));

    std::cout << "wrote recon_*.ply / gt_*.ply\n";
    return 0;
}
```

- [ ] **Step 2: Register the executable in `example2/CMakeLists.txt`**

After the `directional_tsdf_demo` block, add:

```cmake
add_executable(directional_tsdf_eval directional_tsdf_eval.cpp)
target_link_libraries(directional_tsdf_eval PRIVATE Engine::Spatial)
target_compile_definitions(directional_tsdf_eval PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```

- [ ] **Step 3: Build and run — observe the RMSE numbers**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
cmake --build build --target directional_tsdf_eval -j"$(sysctl -n hw.ncpu)"
cd build/example2 && ./directional_tsdf_eval
```
Expected: prints two blocks (sphere, plane) with finite RMSE values. **Record the printed `accuracyRMSE` and `completenessRMSE` for sphere and plane** — Task 4 turns these into regression thresholds. Sanity: with voxelSize=0.1, accuracy RMSE should be on the order of a voxel (≈0.02–0.12); recon_pts should be in the thousands; a wildly larger number or near-zero recon_pts signals a real bug — stop and investigate before Task 4.

- [ ] **Step 4: Sanity-check the PLYs**

```bash
awk 'NR>7 { d=sqrt($1*$1+$2*$2+$3*$3); s+=d; ss+=d*d; n++ } END { m=s/n; print "sphere recon: n="n" mean|p|="m" (expect ~1.0)" }' recon_sphere.ply
```
Expected: mean radius ≈ 1.0 (recon points sit on the unit sphere).

- [ ] **Step 5: Commit**

```bash
git add example2/directional_tsdf_eval.cpp example2/CMakeLists.txt
git commit -m "Add directional_tsdf_eval example (synthetic scan + RMSE report)"
```

---

### Task 4: RMSE regression guard tests (thresholds from Task 3's measured numbers)

**Files:**
- Modify: `test/test_directionalTSDFEval.cpp` (append GPU reconstruction tests)

**Interfaces:**
- Consumes: everything above + `Engine::Core::Context`, `Engine::Spatial::DirectionalTSDF`.

**Threshold policy:** Use the numbers printed by Task 3 Step 3. For each asserted RMSE, set the threshold to `measured × 1.5` rounded up to a clean value, so normal run-to-run variation never trips it but a real regression (2×+) does. The code below uses placeholders `TH_*` — **replace each with the concrete value derived from your Task 3 run before running the test.** (This is the one intentional measure-then-fill step the spec calls out; do not leave `TH_*` symbolic.)

- [ ] **Step 1: Add a shared reconstruction helper + the guard tests**

Append to `test/test_directionalTSDFEval.cpp` (add includes `#include "Engine/Core/Context.h"` and `#include "Engine/Spatial/DirectionalTSDF.h"` at the top):

```cpp
namespace {
    // Scans `surface` on an orbit, reconstructs with DirectionalTSDF, returns recon positions.
    std::vector<Eigen::Vector3f> reconstruct(const Surface &surface, float cameraRadius,
                                             const Eigen::Vector3f &orbitCenter) {
        OrbitParams params;
        params.surfaceSamples = surface.SampleDense(4000);
        params.cameraRadius = cameraRadius;
        params.orbitCenter = orbitCenter;
        auto frames = GenerateOrbitScan(surface, params);

        Engine::Core::Context ctx;
        Engine::Spatial::DirectionalTSDF tsdf;
        tsdf.Build(ctx, 0.1f, 0.3f);
        for (const auto &fr : frames)
            tsdf.Integrate(fr.points, fr.normals, fr.cameraPos, fr.aabbCenterHint);

        std::vector<Eigen::Vector3f> recon;
        recon.reserve(tsdf.PointCloud().size());
        for (const auto &pt : tsdf.PointCloud()) recon.push_back(pt.position);
        return recon;
    }
} // namespace

TEST(DirectionalTSDFEvalTest, SphereAccuracyAndCompletenessWithinTolerance) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    const auto recon = reconstruct(sphere, 3.0f, Eigen::Vector3f(0, 0, 0));
    ASSERT_GT(recon.size(), 1000u); // reconstruction actually produced points

    const auto gt = sphere.SampleDense(8000);
    EXPECT_LT(AccuracyRMSE(recon, sphere), TH_SPHERE_ACC);        // replace TH_SPHERE_ACC
    EXPECT_LT(CompletenessRMSE(gt, recon), TH_SPHERE_COMP);       // replace TH_SPHERE_COMP
}

TEST(DirectionalTSDFEvalTest, PlaneAccuracyWithinTolerance) {
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    const auto recon = reconstruct(plane, 3.0f, Eigen::Vector3f(0, 0, 1.5f));
    ASSERT_GT(recon.size(), 500u);
    EXPECT_LT(AccuracyRMSE(recon, plane), TH_PLANE_ACC);         // replace TH_PLANE_ACC
}
```

- [ ] **Step 2: Fill in the thresholds**

Replace `TH_SPHERE_ACC`, `TH_SPHERE_COMP`, `TH_PLANE_ACC` with `ceil(measured × 1.5 × 100) / 100` from Task 3's printout (a clean 2-decimal value ≥ 1.5× the observed number). Example: if sphere accuracy printed `0.041`, use `0.07f`. Write the literal `float` values directly into the three `EXPECT_LT` lines.

- [ ] **Step 3: Build and run the guard tests**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFEvalTest.*"
```
Expected: 2 tests PASS (thresholds comfortably above the measured values).

- [ ] **Step 4: Full suite regression**

```bash
./build/test/vkspatial_tests 2>&1 | tail -6
```
Expected: all pass except the known pre-existing `WideBVHTest.RadiusMatchesCpuReference`.

- [ ] **Step 5: Commit**

```bash
git add test/test_directionalTSDFEval.cpp
git commit -m "Add DirectionalTSDF reconstruction RMSE regression guards"
```

---

## Completion checklist

- [ ] `Engine::Eval` header-only utils build into both `example2` and `test` with no new CMake target.
- [ ] Pure-CPU metric/surface/scan tests pass (hand-computed values matched).
- [ ] `directional_tsdf_eval` reconstructs sphere + plane, prints finite accuracy/completeness RMSE, exports recon/GT PLYs; sphere recon mean radius ≈ 1.0.
- [ ] Reconstruction RMSE guard tests pass with thresholds derived from measured numbers; empty-recon → +inf trips the guard.
- [ ] Full suite keeps the known baseline (only `WideBVHTest.RadiusMatchesCpuReference` fails).
- [ ] Report the measured sphere/plane RMSE numbers back for review.
