#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <random>
#include <vector>
using Engine::Pipeline::LocalGrid;
using Eigen::Vector3f;

// Brute-force nearest within radius (reference).
static int bruteNearest(const std::vector<Vector3f>& pts, const Vector3f& q, float radius) {
    int best = -1; float bestD2 = radius * radius;
    for (int i = 0; i < (int)pts.size(); ++i) {
        const float d2 = (pts[i] - q).squaredNorm();
        if (d2 < bestD2) { bestD2 = d2; best = i; }
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
        if (b < 0) { EXPECT_LT(g, 0); continue; }
        // Grid and brute may pick different indices only if equidistant; compare distances.
        ASSERT_GE(g, 0);
        EXPECT_NEAR((pts[g] - q).norm(), (pts[b] - q).norm(), 1e-5f);
    }
}

#include "Engine/Core/Context.h"
#include "Engine/Pipeline/Registration/GpuPointToPlaneIcp.h"
#include "Engine/Features/RegistrationTypes.h"

// CPU reference: point-to-plane H,b in T's frame, over grid-NN correspondences, CENTRED on tgt centroid.
static void cpuAccumulate(const std::vector<Vector3f>& src, const Engine::Registration::PointCloud& tgt,
                          const Eigen::Matrix4f& T, float maxCorr,
                          Eigen::Matrix<double,6,6>& H, Eigen::Matrix<double,6,1>& b, int& inliers) {
    Vector3f c = Vector3f::Zero();
    for (auto& q : tgt.points) c += q; c /= float(std::max<size_t>(1, tgt.points.size()));
    std::vector<Vector3f> tc(tgt.points.size());
    for (size_t i=0;i<tc.size();++i) tc[i] = tgt.points[i] - c;
    LocalGrid grid(tc, maxCorr);
    H.setZero(); b.setZero(); inliers = 0;
    const Eigen::Matrix3f R = T.block<3,3>(0,0); const Vector3f t = T.block<3,1>(0,3);
    for (auto& s : src) {
        const Vector3f p = R * (s - c) + t;
        const int qi = grid.Nearest(p, maxCorr);
        if (qi < 0) continue;
        const Vector3f& q = tc[qi]; const Vector3f& n = tgt.normals[qi];
        const float e = (p - q).dot(n);
        Eigen::Matrix<float,6,1> J; J.head<3>() = p.cross(n); J.tail<3>() = n;
        H += (J * J.transpose()).cast<double>(); b += (-J * e).cast<double>(); ++inliers;
    }
}

TEST(GpuIcp, AccumulateMatchesCpu) {
    Engine::Core::Context ctx;
    // A small +Z plane patch as source; a matching plane as target (with +Z normals).
    std::vector<Vector3f> src; Engine::Registration::PointCloud tgt;
    for (int i=-15;i<=15;++i) for (int j=-15;j<=15;++j) {
        src.emplace_back(i*0.02f, j*0.02f, 0.01f);          // 1 cm above the target plane
        tgt.points.emplace_back(i*0.02f, j*0.02f, 0.0f);
        tgt.normals.emplace_back(0,0,1);
    }
    const Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float maxCorr = 0.05f;

    Eigen::Matrix<double,6,6> Hc; Eigen::Matrix<double,6,1> bc; int nc;
    cpuAccumulate(src, tgt, T, maxCorr, Hc, bc, nc);
    ASSERT_GT(nc, 100);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto out = gpu.Accumulate(src, tgt, T, maxCorr);
    EXPECT_EQ(out.inliers, nc);
    EXPECT_TRUE(((out.H - Hc).array().abs() < 1e-2 * (1.0 + Hc.array().abs())).all()) << out.H << "\n---\n" << Hc;
    EXPECT_TRUE(((out.b - bc).array().abs() < 1e-2 * (1.0 + bc.array().abs())).all()) << out.b << "\n---\n" << bc;
}

#include "Engine/Pipeline/Registration/PointToPlaneIcp.h"
#include <Eigen/Geometry>

TEST(GpuIcp, SolveMatchesCpuOnCorner) {
    Engine::Core::Context ctx;
    // A 3-plane corner target (constrains all 6 DoF); source = target perturbed by a small transform.
    Engine::Registration::PointCloud tgt; std::vector<Vector3f> src;
    auto addPlane = [&](const Vector3f& o, const Vector3f& u, const Vector3f& v, const Vector3f& n){
        for (int i=-10;i<=10;++i) for (int j=-10;j<=10;++j) {
            const Vector3f p = o + u*(i*0.03f) + v*(j*0.03f);
            tgt.points.push_back(p); tgt.normals.push_back(n);
        }
    };
    addPlane({0,0,0},{1,0,0},{0,1,0},{0,0,1});
    addPlane({0,0,0},{0,1,0},{0,0,1},{1,0,0});
    addPlane({0,0,0},{1,0,0},{0,0,1},{0,1,0});
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    for (const auto& q : tgt.points) src.push_back(perturb * q); // source is the model, moved

    Engine::Registration::RegistrationParam params; params.maxCorrDist = 0.1f; params.maxIters = 30;
    const auto cpu = Engine::Registration::AlignPointToPlaneIcp(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(cpu.valid);

    Engine::Pipeline::GpuPointToPlaneIcp gpu(ctx);
    const auto g = gpu.Solve(src, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(g.valid);
    // Both should recover ~perturb⁻¹ (align source back onto target). Compare the two poses directly.
    EXPECT_TRUE(((g.T - cpu.T).array().abs() < 5e-3f).all()) << "gpu:\n" << g.T << "\ncpu:\n" << cpu.T;
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
    struct SizeSpec { int M; float spacing; };
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

    for (const auto &sz : sizes) {
        Engine::Registration::PointCloud tgt;
        buildCorner(tgt, sz.M, sz.spacing);
        std::vector<Vector3f> src;
        src.reserve(tgt.points.size());
        for (const auto &q : tgt.points) src.push_back(perturb * q);
        const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();

        // Warmup (excluded from timing): first GPU dispatch at THIS size compiles the shader (once,
        // globally) and (re)allocates buffers for this N; also warms CPU allocations/caches.
        const auto warmGpu = gpu.Solve(src, tgt, I, params);
        const auto warmCpu = Engine::Registration::AlignPointToPlaneIcp(src, tgt, I, params);
        ASSERT_TRUE(warmGpu.valid) << "GPU warmup failed to converge at N=" << tgt.points.size();
        ASSERT_TRUE(warmCpu.valid) << "CPU warmup failed to converge at N=" << tgt.points.size();

        double gpuTotalMs = 0.0, cpuTotalMs = 0.0;
        Engine::Registration::RegistrationResult lastGpu, lastCpu;
        for (int k = 0; k < kRepeats; ++k) {
            const auto g0 = std::chrono::steady_clock::now();
            lastGpu = gpu.Solve(src, tgt, I, params);
            const auto g1 = std::chrono::steady_clock::now();
            gpuTotalMs += std::chrono::duration<double, std::milli>(g1 - g0).count();

            const auto c0 = std::chrono::steady_clock::now();
            lastCpu = Engine::Registration::AlignPointToPlaneIcp(src, tgt, I, params);
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
            << "N=" << tgt.points.size() << " gpu:\n" << lastGpu.T << "\ncpu:\n" << lastCpu.T;

        printf("%10zu | %12.3f | %12.3f | %9.2fx | %d\n", tgt.points.size(), gpuMean, cpuMean,
               cpuMean / gpuMean, int(lastGpu.numInliers));
        fflush(stdout);
    }
    printf("\n");
}
