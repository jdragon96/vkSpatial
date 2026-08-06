#include "Engine/Pipeline/Registration/GpuIcp.h"
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
