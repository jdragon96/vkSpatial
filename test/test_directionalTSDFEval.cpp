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
