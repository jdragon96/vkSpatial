#include "Engine/Eval/RmseMetrics.h"

#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

namespace {

    std::vector<Eigen::Vector3f> randomCloud(int count, unsigned seed, float spread = 1.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-spread, spread);
        std::vector<Eigen::Vector3f> points;
        points.reserve(std::size_t(count));
        for (int i = 0; i < count; ++i) points.emplace_back(u(rng), u(rng), u(rng));
        return points;
    }

} // namespace

// The grid is an ACCELERATION, not an approximation. Asserted against the brute-force scan it
// replaced, because an off-by-one in the ring stopping bound returns a plausible-but-wrong
// distance -- larger than the true nearest, never obviously broken.
TEST(RmseMetrics, GridNearestNeighbourMatchesBruteForce) {
    for (const unsigned seed: {1u, 2u, 7u}) {
        const std::vector<Eigen::Vector3f> from = randomCloud(400, seed);
        const std::vector<Eigen::Vector3f> to = randomCloud(700, seed + 100u);
        EXPECT_FLOAT_EQ(Engine::Eval::NearestNeighbourRMSE(from, to),
                        Engine::Eval::NearestNeighbourRMSEBruteForce(from, to))
                << "seed " << seed;
    }
}

// Clouds that do not overlap put every query outside the target's grid, which is the case the
// ring search has to clamp rather than walk off the end of.
TEST(RmseMetrics, HandlesDisjointClouds) {
    std::vector<Eigen::Vector3f> from = randomCloud(200, 3);
    const std::vector<Eigen::Vector3f> to = randomCloud(200, 4);
    for (Eigen::Vector3f &p: from) p += Eigen::Vector3f(50.0f, -30.0f, 12.0f);

    EXPECT_FLOAT_EQ(Engine::Eval::NearestNeighbourRMSE(from, to),
                    Engine::Eval::NearestNeighbourRMSEBruteForce(from, to));
}

// A degenerate target (every point identical) collapses the grid to one cell; a single-point
// target is the smallest version of the same thing.
TEST(RmseMetrics, HandlesDegenerateTargets) {
    const std::vector<Eigen::Vector3f> from = randomCloud(50, 5);
    const std::vector<Eigen::Vector3f> single{Eigen::Vector3f(0.25f, -0.5f, 0.75f)};
    EXPECT_FLOAT_EQ(Engine::Eval::NearestNeighbourRMSE(from, single),
                    Engine::Eval::NearestNeighbourRMSEBruteForce(from, single));

    const std::vector<Eigen::Vector3f> coincident(64, Eigen::Vector3f(1.0f, 1.0f, 1.0f));
    EXPECT_FLOAT_EQ(Engine::Eval::NearestNeighbourRMSE(from, coincident),
                    Engine::Eval::NearestNeighbourRMSEBruteForce(from, coincident));
}

// The reason the grid exists. icp_quality_diag compares two ~1.9M-point reconstructions in both
// directions; at O(|from|x|to|) that is 3.6e12 distance evaluations per direction and the tool
// ran over seven hours without finishing. 60k x 60k is 3.6e9 -- already minutes brute-force --
// and must complete here in well under a second.
TEST(RmseMetrics, LargeCloudsCompleteQuickly) {
    const std::vector<Eigen::Vector3f> from = randomCloud(60000, 11);
    const std::vector<Eigen::Vector3f> to = randomCloud(60000, 12);

    const auto start = std::chrono::steady_clock::now();
    const float rmse = Engine::Eval::NearestNeighbourRMSE(from, to);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    EXPECT_GT(rmse, 0.0f);
    EXPECT_LT(seconds, 5.0) << "60k x 60k took " << seconds << " s: the grid is not being used";
}
