#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <random>
#include <vector>
using Eigen::Vector3f;
using Engine::Pipeline::LocalGrid;

// Brute-force nearest within radius (reference).
static int bruteNearest(const std::vector<Vector3f> &pts, const Vector3f &q, float radius) {
    int best = -1;
    float bestD2 = radius * radius;
    for (int i = 0; i < (int) pts.size(); ++i) {
        const float d2 = (pts[i] - q).squaredNorm();
        if (d2 < bestD2) {
            bestD2 = d2;
            best = i;
        }
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
        if (b < 0) {
            EXPECT_LT(g, 0);
            continue;
        }
        // Grid and brute may pick different indices only if equidistant; compare distances.
        ASSERT_GE(g, 0);
        EXPECT_NEAR((pts[g] - q).norm(), (pts[b] - q).norm(), 1e-5f);
    }
}

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include "Engine/Pipeline/Registration/RegistrationTypes.h"

// CPU reference: point-to-plane H,b in T's frame, over grid-NN correspondences, CENTRED on tgt centroid.
static void cpuAccumulate(const std::vector<Vector3f> &src, const Engine::Registration::PointCloud &tgt,
                          const Eigen::Matrix4f &T, float maxCorr,
                          Eigen::Matrix<double, 6, 6> &H, Eigen::Matrix<double, 6, 1> &b, int &inliers) {
    Vector3f c = Vector3f::Zero();
    for (auto &q: tgt.points) c += q;
    c /= float(std::max<size_t>(1, tgt.points.size()));
    std::vector<Vector3f> tc(tgt.points.size());
    for (size_t i = 0; i < tc.size(); ++i) tc[i] = tgt.points[i] - c;
    LocalGrid grid(tc, maxCorr);
    H.setZero();
    b.setZero();
    inliers = 0;
    const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
    const Vector3f t = T.block<3, 1>(0, 3);
    for (auto &s: src) {
        const Vector3f p = R * (s - c) + t;
        const int qi = grid.Nearest(p, maxCorr);
        if (qi < 0) continue;
        const Vector3f &q = tc[qi];
        const Vector3f &n = tgt.normals[qi];
        const float e = (p - q).dot(n);
        Eigen::Matrix<float, 6, 1> J;
        J.head<3>() = p.cross(n);
        J.tail<3>() = n;
        H += (J * J.transpose()).cast<double>();
        b += (-J * e).cast<double>();
        ++inliers;
    }
}

TEST(GpuIcp, AccumulateMatchesCpu) {
    Engine::Core::Context ctx;
    // A small +Z plane patch as source; a matching plane as target (with +Z normals).
    std::vector<Vector3f> src;
    Engine::Registration::PointCloud tgt;
    for (int i = -15; i <= 15; ++i)
        for (int j = -15; j <= 15; ++j) {
            src.emplace_back(i * 0.02f, j * 0.02f, 0.01f); // 1 cm above the target plane
            tgt.points.emplace_back(i * 0.02f, j * 0.02f, 0.0f);
            tgt.normals.emplace_back(0, 0, 1);
        }
    const Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float maxCorr = 0.05f;

    Eigen::Matrix<double, 6, 6> Hc;
    Eigen::Matrix<double, 6, 1> bc;
    int nc;
    cpuAccumulate(src, tgt, T, maxCorr, Hc, bc, nc);
    ASSERT_GT(nc, 100);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto out = gpu.Accumulate(src, tgt, T, maxCorr);
    EXPECT_EQ(out.inliers, nc);
    EXPECT_TRUE(((out.H - Hc).array().abs() < 1e-2 * (1.0 + Hc.array().abs())).all()) << out.H << "\n---\n"
                                                                                      << Hc;
    EXPECT_TRUE(((out.b - bc).array().abs() < 1e-2 * (1.0 + bc.array().abs())).all()) << out.b << "\n---\n"
                                                                                      << bc;
}

#include "Engine/Pipeline/Registration/PointToPlaneIcp.h"
#include <Eigen/Geometry>

TEST(GpuIcp, SolveMatchesCpuOnCorner) {
    Engine::Core::Context ctx;
    // A 3-plane corner target (constrains all 6 DoF); source = target perturbed by a small transform.
    Engine::Registration::PointCloud tgt;
    std::vector<Vector3f> src, srcNormals;
    auto addPlane = [&](const Vector3f &o, const Vector3f &u, const Vector3f &v, const Vector3f &n) {
        for (int i = -10; i <= 10; ++i)
            for (int j = -10; j <= 10; ++j) {
                const Vector3f p = o + u * (i * 0.03f) + v * (j * 0.03f);
                tgt.points.push_back(p);
                tgt.normals.push_back(n);
            }
    };
    addPlane({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
    addPlane({0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0});
    addPlane({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    // source is the model, moved -- its true normal at each point is the target's normal there,
    // rotated by the SAME perturbation (normals don't translate); index-aligned with tgt.points/normals.
    for (size_t i = 0; i < tgt.points.size(); ++i) {
        src.push_back(perturb * tgt.points[i]);
        srcNormals.push_back(perturb.rotation() * tgt.normals[i]);
    }

    Engine::Registration::RegistrationParam params;
    params.maxCorrDist = 0.1f;
    params.maxIters = 30;
    const auto cpu = Engine::Registration::AlignPointToPlaneIcp(src, srcNormals, tgt,
                                                                 Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(cpu.valid);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto g = gpu.Solve(src, srcNormals, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(g.valid);
    // Both should recover ~perturb⁻¹ (align source back onto target). Compare the two poses directly.
    EXPECT_TRUE(((g.T - cpu.T).array().abs() < 5e-3f).all()) << "gpu:\n"
                                                             << g.T << "\ncpu:\n"
                                                             << cpu.T;
}

TEST(GpuIcp, ResidualRmseMatchesCpu) {
    Engine::Core::Context ctx;
    // Same 3-plane corner fixture as SolveMatchesCpuOnCorner (constrains all 6 DoF).
    Engine::Registration::PointCloud tgt;
    std::vector<Vector3f> src;
    auto addPlane = [&](const Vector3f &o, const Vector3f &u, const Vector3f &v, const Vector3f &n) {
        for (int i = -10; i <= 10; ++i)
            for (int j = -10; j <= 10; ++j) {
                const Vector3f p = o + u * (i * 0.03f) + v * (j * 0.03f);
                tgt.points.push_back(p);
                tgt.normals.push_back(n);
            }
    };
    addPlane({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
    addPlane({0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0});
    addPlane({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    // This corner is an EXACT rigid map of tgt (no noise), so ICP's Newton iterations converge the
    // point-to-plane residual to ~machine epsilon -- far below the GPU accumulator's fixed-point
    // resolution (SCALE=10000 needs |e| >~ 0.007 to register a nonzero int32; see icp_iterate.comp.glsl).
    // Add small deterministic per-point jitter so the least-squares optimum has a genuine nonzero
    // residual floor (comfortably above that resolution, still well inside maxCorrDist below).
    std::mt19937 jitterRng(7);
    std::uniform_real_distribution<float> jitter(-0.01f, 0.01f);
    for (const auto &q: tgt.points) {
        Vector3f p = perturb * q;
        p += Vector3f(jitter(jitterRng), jitter(jitterRng), jitter(jitterRng));
        src.push_back(p); // source is the model, moved + jittered
    }

    Engine::Registration::RegistrationParam params;
    params.maxCorrDist = 0.1f;

    // No source normals here (this fixture is about the RESIDUAL-RMSE floor, not normal rejection) --
    // {} skips the normal-compatibility check identically on both paths (see AlignPointToPlaneIcp /
    // GpuPointToPlaneIcp::Solve doc comments).
    const auto cpu =
            Engine::Registration::AlignPointToPlaneIcp(src, {}, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(cpu.valid);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto gpuResult = gpu.Solve(src, {}, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(gpuResult.valid);

    EXPECT_GT(cpu.rmse, 0.0f);
    EXPECT_GT(gpuResult.rmse, 0.0f);
    EXPECT_NEAR(gpuResult.rmse, cpu.rmse, 1e-3f) << "gpu " << gpuResult.rmse << " cpu " << cpu.rmse;
}

// ---------------------------------------------------------------------------------------------
// Task 5: headless, deterministic GPU-vs-CPU timing benchmark (DISABLED -- gated, not part of the
// normal suite; run explicitly with --gtest_also_run_disabled_tests --gtest_filter=GpuIcp.DISABLED_*).
//
// Debug builds (-O0) inflate CPU/Eigen 10-100x, so this is only meaningful built RelWithDebInfo
// (build-rel/). It measures ONLY the core ICP solve on equal-N target/source clouds -- the real
// pipeline's GpuIcpTracker additionally crops the model to the frame AABB before calling Solve, an
// extra advantage for the GPU path (smaller upload + grid) that this microbenchmark does not capture.
#include <chrono>
#include <cstdio>

namespace {

    // Dense 3-plane corner target, apex off-origin, extent held ~0.6m across sizes (spacing shrinks as
    // point count grows, per plane count = 3*(2M+1)^2). Normals point along +X/+Y/+Z respectively, so
    // the corner constrains all 6 DoF (matches SolveMatchesCpuOnCorner's fixture, just parameterised).
    void buildCorner(Engine::Registration::PointCloud &tgt, int M, float spacing) {
        const Vector3f apex(0.3f, 0.3f, 0.3f);
        auto addPlane = [&](const Vector3f &u, const Vector3f &v, const Vector3f &n) {
            for (int i = -M; i <= M; ++i)
                for (int j = -M; j <= M; ++j)
                    tgt.points.push_back(apex + u * (i * spacing) + v * (j * spacing)), tgt.normals.push_back(n);
        };
        addPlane({1, 0, 0}, {0, 1, 0}, {0, 0, 1});
        addPlane({0, 1, 0}, {0, 0, 1}, {1, 0, 0});
        addPlane({1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    }

} // namespace

TEST(GpuIcp, DISABLED_BenchmarkVsCpu) {
    Engine::Core::Context ctx;
    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx); // one instance reused across sizes, like the real tracker

    // Target sizes ~5k/20k/80k/200k (3 planes of (2M+1)^2 points each); spacing shrinks with M so the
    // plane extent stays ~0.6m (sub-meter) at every size.
    struct SizeSpec {
        int M;
        float spacing;
    };
    const std::vector<SizeSpec> sizes = {
            {20, 0.015f},     // N ~ 5,043
            {41, 0.00732f},   // N ~ 20,667
            {82, 0.00366f},   // N ~ 81,675
            {129, 0.002326f}, // N ~ 201,243
    };

    // maxCorrDist fixed across sizes: must stay >= the worst-case initial misalignment (translation
    // 2cm + rotation-lever on a ~0.6m corner from Identity, ~3-4cm) to bootstrap correspondences on
    // iteration 1 at every density; 3cm keeps that margin while still tracking point spacing loosely
    // (ratio grows from 2x at the sparsest size to ~13x at the densest -- both CPU's IcpGridNN and the
    // GPU's LocalGrid bucket with cell == maxCorrDist, so this also bounds per-cell occupancy).
    const float maxCorrDist = 0.03f;
    Engine::Registration::RegistrationParam params;
    params.maxCorrDist = maxCorrDist;
    params.maxIters = 20;

    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));

    constexpr int kRepeats = 10;

    printf("\n%10s | %12s | %12s | %10s | %s\n", "N", "GPU mean ms", "CPU mean ms", "speedup", "GPU inliers");
    printf("-----------|--------------|--------------|------------|------------\n");
    fflush(stdout);

    for (const auto &sz: sizes) {
        Engine::Registration::PointCloud tgt;
        buildCorner(tgt, sz.M, sz.spacing);
        std::vector<Vector3f> src;
        src.reserve(tgt.points.size());
        for (const auto &q: tgt.points) src.push_back(perturb * q);
        const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

        // Warmup (excluded from timing): first GPU dispatch at THIS size compiles the shader (once,
        // globally) and (re)allocates buffers for this N; also warms CPU allocations/caches. No source
        // normals -- this benchmark is about raw solve throughput, not robust-correspondence behaviour;
        // {} skips the normal-compatibility check identically on both paths.
        const auto warmGpu = gpu.Solve(src, {}, tgt, I, params);
        const auto warmCpu = Engine::Registration::AlignPointToPlaneIcp(src, {}, tgt, I, params);
        ASSERT_TRUE(warmGpu.valid) << "GPU warmup failed to converge at N=" << tgt.points.size();
        ASSERT_TRUE(warmCpu.valid) << "CPU warmup failed to converge at N=" << tgt.points.size();

        double gpuTotalMs = 0.0, cpuTotalMs = 0.0;
        Engine::Registration::RegistrationResult lastGpu, lastCpu;
        for (int k = 0; k < kRepeats; ++k) {
            const auto g0 = std::chrono::steady_clock::now();
            lastGpu = gpu.Solve(src, {}, tgt, I, params);
            const auto g1 = std::chrono::steady_clock::now();
            gpuTotalMs += std::chrono::duration<double, std::milli>(g1 - g0).count();

            const auto c0 = std::chrono::steady_clock::now();
            lastCpu = Engine::Registration::AlignPointToPlaneIcp(src, {}, tgt, I, params);
            const auto c1 = std::chrono::steady_clock::now();
            cpuTotalMs += std::chrono::duration<double, std::milli>(c1 - c0).count();
        }
        const double gpuMean = gpuTotalMs / kRepeats;
        const double cpuMean = cpuTotalMs / kRepeats;

        // Sanity: both must converge, and to (approximately) the same pose -- the table is the real
        // output of this test, but a silently-broken solve producing garbage-fast timings must fail.
        ASSERT_TRUE(lastGpu.valid) << "GPU failed to converge at N=" << tgt.points.size();
        ASSERT_TRUE(lastCpu.valid) << "CPU failed to converge at N=" << tgt.points.size();
        EXPECT_TRUE(((lastGpu.T - lastCpu.T).array().abs() < 5e-3f).all())
                << "N=" << tgt.points.size() << " gpu:\n"
                << lastGpu.T << "\ncpu:\n"
                << lastCpu.T;

        printf("%10zu | %12.3f | %12.3f | %9.2fx | %d\n", tgt.points.size(), gpuMean, cpuMean,
               cpuMean / gpuMean, int(lastGpu.numInliers));
        fflush(stdout);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------------------------
// Registration-quality plan, Task 2: deterministic perturbation-recovery harness (DISABLED --
// gated, not part of the normal suite; this IS the regression gate every later ICP-quality tier
// must beat/hold). Builds a coarse-voxel-quantized corner ModelSnapshot whose entries carry the
// sub-voxel `tsdf` (so a later tier's sub-voxel target reconstruction is measurable), perturbs a
// source frame by a KNOWN transform, drives the real tracker's Track() (not just Solve, so a later
// tier's target-construction path stays exercisable), and reports recovered-pose error +
// Engine::Eval::NearestNeighbourRMSE (CPU float, no fixed-point floor) as the PRIMARY signal, plus
// the residual `rmse` as a SECONDARY one: Task 1 found the GPU accumulator under-reports residuals
// below ~7mm (fixed-point floor), so this harness does not assert on residual rmse being large.
#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Pipeline/Registration/Tracker.h"

#include <cmath>

namespace {

    // Dense 3-plane corner surface (same construction as SolveMatchesCpuOnCorner / ResidualRmseMatchesCpu
    // above), spacing finer than the harness's voxel size so QuantizeToModel below genuinely coarsens it.
    //
    // kCornerApex is deliberately OFFSET from a voxel-grid multiple (0.05 below), unlike an
    // origin-anchored corner. Why this matters (found in review of the first version of this harness):
    // with the corner anchored at the origin, every plane's constant (normal-axis) coordinate was
    // EXACTLY 0.0 -- itself an exact multiple of ANY voxel size -- so voxel-snapping left that
    // coordinate UNCHANGED and only perturbed the two TANGENTIAL (in-plane) coordinates. Point-to-plane
    // ICP is structurally insensitive to in-plane target error, so entry.tsdf came out exactly 0 for
    // all 1323 entries and the raw-center baseline sat at the numerical floor (measured: transErr
    // ~1e-5, residualRmse 0) -- unable to show any improvement for a later sub-voxel-correction tier to
    // remove, defeating the harness's purpose as a regression gate. Anchoring at an off-grid apex gives
    // every entry a genuine non-zero OUT-OF-PLANE (normal-axis) quantization error instead, so
    // entry.tsdf is meaningfully non-zero and the raw-center baseline sits well above the floor.
    constexpr int kCornerHalfExtent = 10;
    constexpr float kCornerSpacing = 0.03f;
    const Vector3f kCornerApex(0.37f, -0.22f, 0.29f); // NOT a multiple of voxel=0.05 on any axis

    void addCornerPlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals, const Vector3f &u,
                        const Vector3f &v, const Vector3f &n) {
        for (int i = -kCornerHalfExtent; i <= kCornerHalfExtent; ++i)
            for (int j = -kCornerHalfExtent; j <= kCornerHalfExtent; ++j) {
                points.push_back(kCornerApex + u * (i * kCornerSpacing) + v * (j * kCornerSpacing));
                normals.push_back(n);
            }
    }

    // Parallel (points[i], normals[i]) corner surface -- shared by MakeCornerSurfacePoints() and
    // MakeCornerSurfaceNormals() below so the two stay index-aligned by construction.
    void buildCornerSurface(std::vector<Vector3f> &points, std::vector<Vector3f> &normals) {
        addCornerPlane(points, normals, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
        addCornerPlane(points, normals, {0, 1, 0}, {0, 0, 1}, {1, 0, 0});
        addCornerPlane(points, normals, {1, 0, 0}, {0, 0, 1}, {0, 1, 0});
    }

    std::vector<Vector3f> MakeCornerSurfacePoints() {
        std::vector<Vector3f> points, normals;
        buildCornerSurface(points, normals);
        return points;
    }

    std::vector<Vector3f> MakeCornerSurfaceNormals() {
        std::vector<Vector3f> points, normals;
        buildCornerSurface(points, normals);
        return normals;
    }

    // Each of the corner's 3 planes holds ONE coordinate fixed at kCornerApex's value (that plane's own
    // normal axis) and spans the other two from there, so for ANY point p on this surface, the axis of
    // p's SMALLEST |coordinate - kCornerApex| is that plane's constant axis, and the unit vector along
    // it is the plane's normal -- exact for points from addCornerPlane above (that axis's deviation
    // from kCornerApex is exactly 0.0f there); at shared edges two axes tie at 0 and either plane's
    // normal is correct, since the point already lies on both.
    Vector3f cornerPlaneNormal(const Vector3f &p) {
        const Vector3f a = (p - kCornerApex).cwiseAbs();
        if (a.x() <= a.y() && a.x() <= a.z()) return Vector3f(1, 0, 0);
        if (a.y() <= a.x() && a.y() <= a.z()) return Vector3f(0, 1, 0);
        return Vector3f(0, 0, 1);
    }

    // Coarse-voxel-quantize a dense surface into a ModelSnapshot: entry.center is `surfacePoints[i]`
    // snapped to the voxel grid, entry.normal is the corner's plane normal at that point, and
    // entry.tsdf is the signed distance from `center` to the (infinite) plane through kCornerApex with
    // that normal, in truncation units -- so `center - tsdf*truncation*normal` lands back exactly on
    // the true surface (the plane is flat, so this projection is exact for any starting `center`, not
    // an approximation). Because kCornerApex is off the voxel grid (see the block comment above it),
    // `center`'s normal-axis coordinate is snapped AWAY from the true apex coordinate, so this is a
    // genuine non-zero correction -- the invariant a later tier's sub-voxel extraction will exploit.
    Engine::Pipeline::ModelSnapshot QuantizeToModel(const std::vector<Vector3f> &surfacePoints, float voxel,
                                                     float truncation) {
        Engine::Pipeline::ModelSnapshot model;
        model.entries.reserve(surfacePoints.size());
        for (const Vector3f &p: surfacePoints) {
            const Vector3f n = cornerPlaneNormal(p);
            Vector3f center;
            for (int axis = 0; axis < 3; ++axis) center[axis] = std::round(p[axis] / voxel) * voxel;
            const float signedDistance = n.dot(center - kCornerApex); // plane through kCornerApex
            Engine::Spatial::AdvancedEntry entry;
            entry.center = center;
            entry.direction = 0;
            entry.tsdf = signedDistance / truncation;
            entry.weight = 1.0f;
            entry.normal = n;
            entry.firstFrame = 0;
            model.entries.push_back(entry);
        }
        return model;
    }

} // namespace

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

    // Registration-quality plan, Task 3 (Tier 1): the tracker now targets the sub-voxel surface point
    // (center - tsdf*truncationDistance*normal) instead of the quantized voxel center, so the recovered
    // pose should stop absorbing the voxel-quantization offset. Task 2's raw-center baseline measured
    // transErr 0.03001 / reconNnRmse 0.02237; both thresholds sit safely below that baseline but above
    // the sub-voxel result actually achieved (~1e-3 or lower), so this fails if the sub-voxel target
    // construction regresses back toward raw centers.
    EXPECT_LT(recoveredTranslationError, 0.01f)
            << "sub-voxel target should beat the Task 2 raw-center baseline (transErr 0.03001)";
    EXPECT_LT(reconRmse, 0.01f)
            << "sub-voxel target should beat the Task 2 raw-center baseline (reconNnRmse 0.02237)";
}

// ---------------------------------------------------------------------------------------------
// Registration-quality plan, Task 4 (Tier 2): noisy/outlier fixture. The harness above is CLEAN (an
// exact rigid map of the model, no sensor noise), so it cannot show any benefit from robust
// weighting -- this test contaminates the same corner fixture with Gaussian sensor noise on some
// points plus a fraction of gross outliers (both a larger positional offset AND a corrupted/negated
// normal -- a realistic depth-discontinuity artifact), then solves it TWICE with the GPU path:
// once with Huber weighting + normal rejection disabled (huge huberScale, empty sourceNormals --
// reproduces the pre-Tier-2/Task 1-3 behaviour) and once with them enabled at the values the real
// trackers use (huberScale = voxel, default normalCompatibilityCosine). DISABLED -- gated like the
// harness above, not part of the normal suite; this is a MEASUREMENT test (records both numbers).
TEST(GpuIcp, DISABLED_RegistrationQualityHarnessNoisyRobustness) {
    Engine::Core::Context ctx;
    const float voxel = 0.05f, truncation = 0.15f;
    std::vector<Eigen::Vector3f> trueSurface = MakeCornerSurfacePoints(); // dense corner, sub-voxel
    std::vector<Eigen::Vector3f> trueNormals = MakeCornerSurfaceNormals(); // parallel to trueSurface
    Engine::Pipeline::ModelSnapshot model = QuantizeToModel(trueSurface, voxel, truncation);
    model.voxel = voxel;
    model.truncationDistance = truncation;

    Eigen::Isometry3f knownPerturbation = Eigen::Isometry3f::Identity();
    knownPerturbation.translate(Eigen::Vector3f(0.02f, -0.015f, 0.01f));
    knownPerturbation.rotate(Eigen::AngleAxisf(0.03f, Eigen::Vector3f::UnitZ()));

    // Contaminate the clean (knownPerturbation-only) source frame: every 4th point gets small
    // Gaussian sensor noise (sigma 1cm -- Huber's target: inflates the residual moderately, still
    // findable within maxCorrDist). Every 15th point ALSO gets a larger (but still inside
    // maxCorrDist=2*voxel=0.1, so the PRE-EXISTING hard distance gate alone cannot drop it) random
    // offset AND a negated normal -- a realistic depth-edge artifact, and exactly what
    // normal-compatibility rejection exists to catch.
    std::mt19937 noiseRng(42);
    std::normal_distribution<float> gaussianNoise(0.0f, 0.01f);
    std::uniform_real_distribution<float> outlierOffsetMag(0.04f, 0.08f);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    Engine::Pipeline::Frame frame;
    int numOutliers = 0, numNoisy = 0;
    for (size_t i = 0; i < trueSurface.size(); ++i) {
        Eigen::Vector3f p = knownPerturbation * trueSurface[i];
        Eigen::Vector3f n = knownPerturbation.rotation() * trueNormals[i];
        if (i % 15 == 0) {
            const Eigen::Vector3f randomDir =
                    Eigen::Vector3f(unit(noiseRng), unit(noiseRng), unit(noiseRng)).normalized();
            p += randomDir * outlierOffsetMag(noiseRng);
            n = -n; // corrupted normal: no longer compatible with the true target normal
            ++numOutliers;
        } else if (i % 4 == 0) {
            p += Eigen::Vector3f(gaussianNoise(noiseRng), gaussianNoise(noiseRng), gaussianNoise(noiseRng));
            ++numNoisy;
        }
        frame.pts.push_back(p);
        frame.nrm.push_back(n);
    }

    // Same target construction the real trackers use (uncropped -- the model here is small enough
    // that cropping is unnecessary): sub-voxel surface point per entry.
    Engine::Registration::PointCloud tgt;
    tgt.points.reserve(model.entries.size());
    tgt.normals.reserve(model.entries.size());
    for (const auto &entry: model.entries) {
        tgt.points.push_back(entry.center - entry.tsdf * truncation * entry.normal);
        tgt.normals.push_back(entry.normal);
    }

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);

    // Tier 1 baseline: Huber weighting + normal rejection both effectively OFF (huge huberScale =>
    // robustWeight == 1 always; empty sourceNormals => rejection skipped entirely), everything else
    // identical -- reproduces the pre-Tier-2 (Task 1-3) behaviour this task must beat.
    Engine::Registration::RegistrationParam nonRobustParams;
    nonRobustParams.maxCorrDist = 2.0f * voxel;
    nonRobustParams.huberScale = 1e6f;
    const auto nonRobust =
            gpu.Solve(frame.pts, {}, tgt, Eigen::Isometry3f::Identity().matrix(), nonRobustParams);

    // Tier 2: robust weighting + normal rejection at the values the real trackers set (huberScale =
    // model voxel; default normalCompatibilityCosine, ~60deg).
    Engine::Registration::RegistrationParam robustParams;
    robustParams.maxCorrDist = 2.0f * voxel;
    robustParams.huberScale = voxel;
    const auto robust =
            gpu.Solve(frame.pts, frame.nrm, tgt, Eigen::Isometry3f::Identity().matrix(), robustParams);

    ASSERT_TRUE(nonRobust.valid);
    ASSERT_TRUE(robust.valid);

    auto measure = [&](const Engine::Registration::RegistrationResult &r, float &transErr, float &reconRmse) {
        const Eigen::Isometry3f pose(r.T);
        const Eigen::Isometry3f error = pose * knownPerturbation; // should be ~identity
        transErr = error.translation().norm();
        // reconRmse: apply the recovered pose to the CLEAN (uncontaminated) perturbed surface, NOT the
        // noisy/outlier frame.pts -- a noisy/outlier point stays ~its own injected offset away from the
        // true surface under ANY rigid correction, so measuring against frame.pts would have the
        // contamination's own footprint dominate the metric regardless of pose quality, masking the
        // very effect under test. This isolates POSE quality, consistent with transErr.
        std::vector<Eigen::Vector3f> alignedClean;
        alignedClean.reserve(trueSurface.size());
        for (const auto &s: trueSurface) alignedClean.push_back(pose * (knownPerturbation * s));
        reconRmse = Engine::Eval::NearestNeighbourRMSE(alignedClean, trueSurface);
    };
    float transErrNonRobust = 0.0f, reconRmseNonRobust = 0.0f, transErrRobust = 0.0f, reconRmseRobust = 0.0f;
    measure(nonRobust, transErrNonRobust, reconRmseNonRobust);
    measure(robust, transErrRobust, reconRmseRobust);

    std::printf("[harness-noisy] outliers=%d noisy=%d/%zu | non-robust transErr %.5f reconNnRmse %.5f | "
                "robust transErr %.5f reconNnRmse %.5f\n",
                numOutliers, numNoisy, trueSurface.size(), transErrNonRobust, reconRmseNonRobust,
                transErrRobust, reconRmseRobust);

    // Registration-quality plan, Task 4 (Tier 2): on a fixture contaminated with sensor noise + gross
    // outliers, Huber weighting + normal rejection must measurably beat the non-robust baseline.
    EXPECT_LT(transErrRobust, transErrNonRobust)
            << "robust weighting + normal rejection should beat the non-robust baseline on a noisy fixture";
    EXPECT_LT(reconRmseRobust, reconRmseNonRobust)
            << "robust weighting + normal rejection should beat the non-robust baseline on a noisy fixture";
}
