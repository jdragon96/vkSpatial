// FPFH descriptor + shared oriented-point extraction (Engine::Spatial).
// CPU-only correctness (no GPU) + integration through SimpleTSDF / DirectionalTSDF.

#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalTSDF.h"
#include "Engine/Spatial/FPFH.h"
#include "Engine/Spatial/NeighborQuery.h"
#include "Engine/Core/OrientedPointCloud.h"
#include "TSDF/Backends/SimpleTSDF.h"

#include <Eigen/Geometry>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <set>
#include <vector>

using namespace Engine::Spatial;

namespace {

    // Random oriented cloud in [0,1]^3 with unit normals (deterministic seed).
    OrientedPointCloud makeRandomCloud(size_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::uniform_real_distribution<float> p(0.0f, 1.0f);
        OrientedPointCloud c;
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
    const OrientedPointCloud cloud = makeRandomCloud(400, 1);
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
    const OrientedPointCloud cloud = makeRandomCloud(300, 2);
    const FpfhConfig cfg{0.4f};
    const auto base = ComputeFPFH(cloud, cfg);

    // Apply a random proper rotation + translation to points AND normals.
    const Eigen::Matrix3f R =
            Eigen::AngleAxisf(1.1f, Eigen::Vector3f(0.3f, -0.7f, 0.65f).normalized()).toRotationMatrix();
    const Eigen::Vector3f t(5.0f, -2.0f, 3.0f);
    OrientedPointCloud moved;
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
    const OrientedPointCloud cloud = makeRandomCloud(250, 3);
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

// ── GPU: SimpleTSDF → shared oriented cloud → FPFH ──────────────────────────
TEST(FpfhTsdfTest, SimpleTSDFExtractAndDescribe) {
    Engine::Core::Context ctx;
    SimpleTSDF tsdf;
    tsdf.Build(ctx, /*voxelSize=*/0.1f, /*truncation=*/0.3f);

    // Fibonacci-sphere surface points (radius 1), integrated from an outside camera.
    std::vector<Eigen::Vector3f> pts;
    const int N = 1500;
    const float golden = float(M_PI) * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < N; ++i) {
        const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(N);
        const float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float a = golden * float(i);
        pts.emplace_back(r * std::cos(a), y, r * std::sin(a));
    }
    tsdf.Integrate(pts, Eigen::Vector3f(0, 0, 3));

    const OrientedPointCloud cloud = tsdf.ExtractPointCloud();
    ASSERT_FALSE(cloud.empty()) << "SimpleTSDF extracted no surface";
    EXPECT_EQ(cloud.points.size(), cloud.normals.size());
    for (const auto &nrm : cloud.normals)
        EXPECT_NEAR(nrm.norm(), 1.0f, 1e-3f); // welded normals are unit length

    const auto sig = ComputeFPFH(cloud, FpfhConfig{0.25f});
    ASSERT_EQ(sig.size(), cloud.size());
    for (const auto &s : sig)
        for (int blk = 0; blk < 3; ++blk) {
            const float bs = blockSum(s, blk);
            EXPECT_TRUE(bs == 0.0f || std::fabs(bs - 100.0f) < 1e-2f); // normalised or isolated
        }
}

// ── GPU: DirectionalTSDF oriented-cloud adapter mirrors PointCloud() ─────────
TEST(FpfhTsdfTest, DirectionalTSDFOrientedCloudMatchesPointCloud) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, /*poolCapacity=*/4096);

    // Small cloud (N<1000, below the known Engine::Core large-N issue).
    std::vector<Eigen::Vector3f> pts, nms;
    const int N = 600;
    const float golden = float(M_PI) * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < N; ++i) {
        const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(N);
        const float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float a = golden * float(i);
        const Eigen::Vector3f p(r * std::cos(a), y, r * std::sin(a));
        pts.push_back(p);
        nms.push_back(p.normalized()); // outward normal
    }
    tsdf.Integrate(pts, nms, Eigen::Vector3f(0, 0, 3), Eigen::Vector3f::Zero());

    const OrientedPointCloud cloud = tsdf.ExtractOrientedCloud();
    ASSERT_EQ(cloud.points.size(), tsdf.PointCloud().size());
    ASSERT_EQ(cloud.normals.size(), cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        EXPECT_EQ(cloud.points[i], tsdf.PointCloud()[i].position);
        EXPECT_EQ(cloud.normals[i], tsdf.PointCloud()[i].normal);
    }

    if (!cloud.empty()) {
        const auto sig = ComputeFPFH(cloud, FpfhConfig{0.25f});
        EXPECT_EQ(sig.size(), cloud.size());
    }
}
