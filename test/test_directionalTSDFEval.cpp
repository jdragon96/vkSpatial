#include <gtest/gtest.h>

#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "Engine/Eval/ScanSampler.h"
#include "Engine/Eval/SyntheticSurface.h"
#include "Engine/Spatial/DirectionalTSDF.h"

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

TEST(SyntheticSurfaceTest, SphereDistanceAndDenseSampling) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    EXPECT_NEAR(sphere.Distance(Eigen::Vector3f(2, 0, 0)), 1.0f, 1e-6f);    // outside
    EXPECT_NEAR(sphere.Distance(Eigen::Vector3f(0.5f, 0, 0)), 0.5f, 1e-6f); // inside
    auto dense = sphere.SampleDense(2000);
    ASSERT_EQ(dense.size(), 2000u);
    for (const auto &p : dense) EXPECT_NEAR(p.norm(), 1.0f, 1e-4f); // on the unit sphere
}

TEST(ScanSamplerTest, SphereScanCoversAllSamplesAcrossViews) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    OrbitParams params;
    params.surfaceSamples = sphere.SampleDense(300); // small: the coverage check below is O(frames×samples²)
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

// Thresholds derived from the directional_tsdf_eval example run (measured × ~2 headroom):
//   sphere accuracy 0.0228 → 0.05, sphere completeness 0.0464 → 0.08, plane accuracy 0.0084 → 0.02.
TEST(DirectionalTSDFEvalTest, SphereAccuracyAndCompletenessWithinTolerance) {
    SphereSurface sphere(Eigen::Vector3f(0, 0, 0), 1.0f);
    const auto recon = reconstruct(sphere, 3.0f, Eigen::Vector3f(0, 0, 0));
    ASSERT_GT(recon.size(), 1000u); // reconstruction actually produced points

    const auto gt = sphere.SampleDense(8000);
    EXPECT_LT(AccuracyRMSE(recon, sphere), 0.05f);
    EXPECT_LT(CompletenessRMSE(gt, recon), 0.08f);
}

TEST(DirectionalTSDFEvalTest, PlaneAccuracyWithinTolerance) {
    PlaneSurface plane(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 1.0f, 1.0f);
    const auto recon = reconstruct(plane, 3.0f, Eigen::Vector3f(0, 0, 1.5f));
    ASSERT_GT(recon.size(), 500u);
    EXPECT_LT(AccuracyRMSE(recon, plane), 0.02f);
}
