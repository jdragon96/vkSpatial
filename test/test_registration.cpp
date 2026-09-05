#include "Features/Downsample.h"
#include "Features/FeatureMatching.h"
#include "Features/Fpfh.h"
#include "GlobalRegistration/GlobalRegistrationPipeline.h"
#include "Features/RegistrationTypes.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <random>
using namespace Registration;

TEST(Registration, VoxelDownsampleReducesAndKeepsExtent) {
    PointCloud in;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j) {
            in.points.emplace_back(i * 0.25f, j * 0.25f, 0.0f); // dense 10x10mm plane, 0.25mm spacing
            in.normals.emplace_back(0, 0, 1);
        }
    PointCloud out = Features::DownsampleVoxel(in, 1.0f); // 1mm cells → ~10x10 = ~100 pts
    EXPECT_LT(out.points.size(), in.points.size());
    EXPECT_GT(out.points.size(), 50u);
    EXPECT_EQ(out.normals.size(), out.points.size());
    // normals preserved (all +Z)
    for (auto &n: out.normals) EXPECT_NEAR(n.z(), 1.0f, 1e-3f);
}

static Registration::PointCloud makeSphere(int n, float r) {
    Registration::PointCloud c;
    for (int i = 0; i < n; ++i) {
        float a = 2.399963f * i, z = 1.0f - 2.0f * (i + 0.5f) / n;
        float rr = std::sqrt(std::max(0.0f, 1 - z * z));
        Eigen::Vector3f d(rr * std::cos(a), rr * std::sin(a), z);
        c.points.push_back(r * d);
        c.normals.push_back(d); // outward normals
    }
    return c;
}

TEST(Registration, FpfhIsApproximatelyRotationInvariant) {
    auto s = makeSphere(600, 20.0f);
    Eigen::Matrix3f R = Eigen::AngleAxisf(0.7f, Eigen::Vector3f(0.3f, 0.8f, 0.5f).normalized()).toRotationMatrix();
    Registration::PointCloud sr = s;
    for (auto &p: sr.points) p = R * p;
    for (auto &nrm: sr.normals) nrm = R * nrm;
    auto f0 = Features::ComputeFpfh(s, 60.0f, 100.0f);
    auto f1 = Features::ComputeFpfh(sr, 60.0f, 100.0f);
    // point i maps to point i under R (same ordering), so descriptors should be close
    double maxdiff = 0;
    for (size_t i = 0; i < f0.size(); ++i) maxdiff = std::max<double>(maxdiff, (f0[i] - f1[i]).norm());
    // Tuned from the measured value (~1.2e-7, floating-point-level agreement for a correct
    // implementation) vs. deliberately-broken Darboux-frame variants tried during
    // development, which measured ~0.033-0.039 (mean per-point descriptor norm ~0.65-1.2).
    // 0.01 sits ~4 orders of magnitude above the measured noise floor and ~3x below the
    // smallest broken-implementation value observed, so it is small relative to descriptor
    // magnitude yet still discriminating.
    EXPECT_LT(maxdiff, 0.01) << "FPFH not rotation-invariant enough (max L2 " << maxdiff << ")";
}

TEST(Registration, MatchRecoversIdentityCorrespondencesUnderRotation) {
    auto s = makeSphere(500, 20.0f);
    Eigen::Matrix3f R = Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    Registration::PointCloud sr = s;
    for (auto &p: sr.points) p = R * p;
    for (auto &n: sr.normals) n = R * n;
    auto fs = Features::ComputeFpfh(s, 60.0f, 100.0f);
    auto ft = Features::ComputeFpfh(sr, 60.0f, 100.0f);
    auto corr = Features::MatchFeatures(fs, ft);
    // most correspondences should be i→i (descriptor space is symmetric under R)
    int selfMatches = 0;
    for (auto &c: corr)
        if (c.srcIdx == c.tgtIdx) ++selfMatches;
    EXPECT_GT(corr.size(), 100u);
    EXPECT_GT(double(selfMatches) / corr.size(), 0.5) << selfMatches << "/" << corr.size();
}

TEST(Registration, RansacRecoversKnownTransform) {
    auto tgt = makeSphere(600, 20.0f); // "model"
    // bumpy sphere so FPFH isn't degenerate: perturb radius by a hash
    for (size_t i = 0; i < tgt.points.size(); ++i) tgt.points[i] *= (1.0f + 0.15f * std::sin(0.7f * i));
    Eigen::Matrix3f Rgt = Eigen::AngleAxisf(0.6f, Eigen::Vector3f(0.2f, 0.7f, 0.6f).normalized()).toRotationMatrix();
    Eigen::Vector3f tgt_t(8.0f, -5.0f, 3.0f);
    Registration::PointCloud src = tgt; // src = model moved by Tgt
    for (size_t i = 0; i < src.points.size(); ++i) {
        src.points[i] = Rgt * tgt.points[i] + tgt_t;
        src.normals[i] = Rgt * tgt.normals[i];
    }
    Registration::RegistrationConfig cfg;
    cfg.voxelSize = 2.0f;
    auto res = Registration::EstimateRansac(src, tgt, cfg); // aligns src→tgt ⇒ T ≈ [Rgt|tgt_t]^-1
    ASSERT_TRUE(res.valid);
    Eigen::Matrix4f Tgt = Eigen::Matrix4f::Identity();
    Tgt.block<3, 3>(0, 0) = Rgt;
    Tgt.block<3, 1>(0, 3) = tgt_t;
    Eigen::Matrix4f err = res.T * Tgt; // should be ≈ identity
    float rotErr = Eigen::AngleAxisf(Eigen::Matrix3f(err.block<3, 3>(0, 0))).angle();
    float trErr = err.block<3, 1>(0, 3).norm();
    EXPECT_LT(rotErr, 0.1f) << "rot err rad"; // ~6°
    EXPECT_LT(trErr, 3.0f) << "trans err mm"; // coarse RANSAC tolerance
}

namespace {
    // Shared "known SE(3) transform" fixture for the Ceres-refine and outlier-robustness
    // tests below -- same shape as the fixture inlined in RansacRecoversKnownTransform above
    // (factored out per Task 5's brief; that existing test is left untouched/unmodified).
    // tgt = "model" (bumpy sphere); src = tgt moved by (Rgt, tgt_t). Registering src->tgt
    // should recover T ≈ [Rgt|tgt_t]^-1, i.e. res.T * Tgt ≈ Identity.
    void MakeKnownTransformFixture(Registration::PointCloud &src, Registration::PointCloud &tgt,
                                   Eigen::Matrix4f &Tgt) {
        tgt = makeSphere(600, 20.0f);
        for (size_t i = 0; i < tgt.points.size(); ++i) tgt.points[i] *= (1.0f + 0.15f * std::sin(0.7f * i));
        Eigen::Matrix3f Rgt =
                Eigen::AngleAxisf(0.6f, Eigen::Vector3f(0.2f, 0.7f, 0.6f).normalized()).toRotationMatrix();
        Eigen::Vector3f tgt_t(8.0f, -5.0f, 3.0f);
        src = tgt;
        for (size_t i = 0; i < src.points.size(); ++i) {
            src.points[i] = Rgt * tgt.points[i] + tgt_t;
            src.normals[i] = Rgt * tgt.normals[i];
        }
        Tgt = Eigen::Matrix4f::Identity();
        Tgt.block<3, 3>(0, 0) = Rgt;
        Tgt.block<3, 1>(0, 3) = tgt_t;
    }
} // namespace

TEST(Registration, CeresRefineTightensRecovery) {
    Registration::PointCloud src, tgt;
    Eigen::Matrix4f Tgt;
    MakeKnownTransformFixture(src, tgt, Tgt);
    Registration::RegistrationConfig cfg;
    cfg.voxelSize = 2.0f;
    auto coarse = Registration::EstimateRansac(src, tgt, cfg);
    auto refined = Registration::Estimate(src, tgt, cfg);
    ASSERT_TRUE(refined.valid);
    Eigen::Matrix4f errC = coarse.T * Tgt, errR = refined.T * Tgt;
    float rotC = Eigen::AngleAxisf(Eigen::Matrix3f(errC.block<3, 3>(0, 0))).angle();
    float rotR = Eigen::AngleAxisf(Eigen::Matrix3f(errR.block<3, 3>(0, 0))).angle();
    EXPECT_LT(rotR, 0.03f) << "refined rot err rad (~1.7°)";
    EXPECT_LE(rotR, rotC + 1e-4f) << "Ceres refine should not worsen the coarse estimate";
}

// Task-4's RansacRecoversKnownTransform fixture is noise-free (an exact rigid transform of
// tgt's own points/normals), so every FPFH match is correct and RANSAC's outlier rejection is
// never actually exercised (fitness=1.0). This test corrupts a fraction of src *points*
// (gross random position + random normal, before registering) so a chunk of the resulting
// FPFH matches are wrong, then checks two things:
//   1. Registration::Estimate (RANSAC + Ceres) still recovers the known transform.
//   2. A "no-RANSAC" baseline -- SolveRigidUmeyama over ALL matched correspondences from the
//      SAME corrupted cloud pair, with no outlier rejection at all -- does NOT recover it.
// (2) is the discriminating half: it proves RANSAC's max-inlier search (not just Ceres's
// Cauchy loss) is doing real work, by showing what happens without it.
TEST(Registration, EstimateRecoversUnderOutlierCorruptionButNaiveBaselineFails) {
    Registration::PointCloud src, tgt;
    Eigen::Matrix4f Tgt;
    MakeKnownTransformFixture(src, tgt, Tgt);

    // Corrupt a fraction of src points with gross outliers (random position well outside the
    // model's ~20mm-radius extent + random unit normal). Deterministic RNG seed.
    std::mt19937 rng(777u);
    std::uniform_real_distribution<float> coordDist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> unitDist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> frac01(0.0f, 1.0f);
    constexpr float kOutlierFrac = 0.40f;
    int numCorrupted = 0;
    for (size_t i = 0; i < src.points.size(); ++i) {
        if (frac01(rng) < kOutlierFrac) {
            src.points[i] = Eigen::Vector3f(coordDist(rng), coordDist(rng), coordDist(rng));
            Eigen::Vector3f n(unitDist(rng), unitDist(rng), unitDist(rng));
            if (n.norm() < 1e-6f) n = Eigen::Vector3f::UnitZ();
            src.normals[i] = n.normalized();
            ++numCorrupted;
        }
    }
    ASSERT_GT(numCorrupted, 0);

    Registration::RegistrationConfig cfg;
    cfg.voxelSize = 2.0f;

    // (1) RANSAC + Ceres should still recover the transform.
    auto refined = Registration::Estimate(src, tgt, cfg);
    ASSERT_TRUE(refined.valid) << numCorrupted << "/" << src.points.size() << " src points corrupted";
    Eigen::Matrix4f errR = refined.T * Tgt;
    float rotR = Eigen::AngleAxisf(Eigen::Matrix3f(errR.block<3, 3>(0, 0))).angle();
    float trR = errR.block<3, 1>(0, 3).norm();
    EXPECT_LT(rotR, 0.1f) << "RANSAC+Ceres rot err rad under outlier corruption (" << numCorrupted
                          << " corrupted)";
    EXPECT_LT(trR, 3.0f) << "RANSAC+Ceres trans err mm under outlier corruption";

    // (2) No-RANSAC baseline: re-run downsample/FPFH/match on the SAME corrupted clouds, but
    // Umeyama-fit ALL matched correspondences directly (no RANSAC outlier rejection).
    const auto srcDs = Features::DownsampleVoxel(src, cfg.voxelSize);
    const auto tgtDs = Features::DownsampleVoxel(tgt, cfg.voxelSize);
    const float normalRadius = cfg.normalRadiusGain * cfg.voxelSize;
    const float fpfhRadius = cfg.fpfhRadiusGain * cfg.voxelSize;
    auto srcF = Features::ComputeFpfh(srcDs, normalRadius, fpfhRadius);
    auto tgtF = Features::ComputeFpfh(tgtDs, normalRadius, fpfhRadius);
    auto corr = Features::MatchFeatures(srcF, tgtF, 0.95f, cfg.numMaxCorr);
    ASSERT_GE(corr.size(), 3u);

    std::vector<Eigen::Vector3f> allSrc, allDst;
    allSrc.reserve(corr.size());
    allDst.reserve(corr.size());
    for (auto &c: corr) {
        allSrc.push_back(srcDs.points[c.srcIdx]);
        allDst.push_back(tgtDs.points[c.tgtIdx]);
    }
    Eigen::Matrix4f baselineT = Registration::SolveRigidUmeyama(allSrc, allDst);
    Eigen::Matrix4f errB = baselineT * Tgt;
    float rotB = Eigen::AngleAxisf(Eigen::Matrix3f(errB.block<3, 3>(0, 0))).angle();
    float trB = errB.block<3, 1>(0, 3).norm();

    EXPECT_GT(rotB, 0.3f) << "no-RANSAC baseline should be dragged far from the truth by outlier "
                             "correspondences (rot err rad="
                          << rotB << ", trans err mm=" << trB << ") -- if this fails, RANSAC's "
                                                                 "max-inlier rejection isn't actually needed for this fixture";
}
