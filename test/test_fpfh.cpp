// FPFH descriptor + shared oriented-point extraction (Engine::Spatial).
// CPU-only correctness. The GPU integration tests went with the TSDF backends they drove.

#include "Engine/Core/Context.h"
#include "Features/FpfhSignature.h"
#include "BVH/NeighborQuery.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <Eigen/Geometry>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <vector>

using namespace Engine::Features;
using namespace Engine::Spatial; // FPFH stayed behind in Engine::Spatial

namespace {

    // Random oriented cloud in [0,1]^3 with unit normals (deterministic seed).
    Engine::Core::OrientedPointCloud makeRandomCloud(size_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::uniform_real_distribution<float> p(0.0f, 1.0f);
        Engine::Core::OrientedPointCloud c;
        c.points.reserve(n);
        c.normals.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            c.points.emplace_back(p(rng), p(rng), p(rng));
            Eigen::Vector3f nrm(u(rng), u(rng), u(rng));
            if (nrm.norm() < 1e-4f) nrm = Eigen::Vector3f(0, 0, 1);
            c.normals.push_back(nrm.normalized());
        }
        return c;
    }

    float blockSum(const FpfhSignature &s, int block) {
        float acc = 0.0f;
        for (int k = 0; k < FPFH_BINS; ++k) acc += s.hist[block * FPFH_BINS + k];
        return acc;
    }

    float l1(const FpfhSignature &a, const FpfhSignature &b) {
        float d = 0.0f;
        for (int c = 0; c < FPFH_DIM; ++c) d += std::fabs(a.hist[c] - b.hist[c]);
        return d;
    }

} // namespace

// ── CPU: neighbourhood grid equals brute force ──────────────────────────────
TEST(FpfhNeighborhoodTest, GridMatchesBruteForce) {
    const Engine::Core::OrientedPointCloud cloud = makeRandomCloud(400, 1);
    const float radius = 0.25f;
    CpuGridNeighborhood grid(cloud.points, radius);

    std::vector<uint32_t> idx;
    std::vector<float> dist;
    for (size_t q = 0; q < cloud.points.size(); q += 7) {
        grid.Radius(cloud.points[q], radius, idx, dist);
        std::set<uint32_t> gridSet(idx.begin(), idx.end());

        std::set<uint32_t> bruteSet;
        for (uint32_t i = 0; i < cloud.points.size(); ++i)
            if ((cloud.points[i] - cloud.points[q]).norm() <= radius) bruteSet.insert(i);

        EXPECT_EQ(gridSet, bruteSet) << "mismatch at query " << q;
        // Distances must be consistent with reported indices.
        for (size_t m = 0; m < idx.size(); ++m)
            EXPECT_NEAR(dist[m], (cloud.points[idx[m]] - cloud.points[q]).norm(), 1e-5f);
    }
}

// ── CPU: FPFH is invariant to rigid transform (its defining property) ────────
TEST(FpfhTest, RotationTranslationInvariant) {
    const Engine::Core::OrientedPointCloud cloud = makeRandomCloud(300, 2);
    const FpfhConfig cfg{0.4f};
    const auto base = ComputeFPFH(cloud, cfg);

    // Apply a random proper rotation + translation to points AND normals.
    const Eigen::Matrix3f R =
            Eigen::AngleAxisf(1.1f, Eigen::Vector3f(0.3f, -0.7f, 0.65f).normalized()).toRotationMatrix();
    const Eigen::Vector3f t(5.0f, -2.0f, 3.0f);
    Engine::Core::OrientedPointCloud moved;
    moved.points.reserve(cloud.size());
    moved.normals.reserve(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        moved.points.push_back(R * cloud.points[i] + t);
        moved.normals.push_back(R * cloud.normals[i]);
    }
    const auto after = ComputeFPFH(moved, cfg);

    ASSERT_EQ(base.size(), after.size());
    double meanL1 = 0.0;
    for (size_t i = 0; i < base.size(); ++i) meanL1 += l1(base[i], after[i]);
    meanL1 /= double(base.size());
    // Features are exactly invariant; only rare bin-boundary float flips remain → mean L1 ~ 0.
    // A non-invariant implementation would score in the tens here.
    EXPECT_LT(meanL1, 1.0) << "FPFH not rigid-invariant (mean L1 = " << meanL1 << ")";
}

// ── CPU: determinism + PCL-style per-block normalisation ────────────────────
TEST(FpfhTest, DeterministicAndBlocksNormalized) {
    const Engine::Core::OrientedPointCloud cloud = makeRandomCloud(250, 3);
    const FpfhConfig cfg{0.4f}; // large enough that every point has neighbours
    const auto a = ComputeFPFH(cloud, cfg);
    const auto b = ComputeFPFH(cloud, cfg);

    ASSERT_EQ(a.size(), cloud.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(l1(a[i], b[i]), 0.0f) << "non-deterministic at " << i; // identical input → identical output
        for (int blk = 0; blk < 3; ++blk)
            EXPECT_NEAR(blockSum(a[i], blk), 100.0f, 1e-2f) << "block " << blk << " not normalised at " << i;
    }
}
