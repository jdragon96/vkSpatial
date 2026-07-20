#include "Engine/Registration/Downsample.h"
#include "Engine/Registration/FeatureMatching.h"
#include "Engine/Registration/Fpfh.h"
#include "Engine/Registration/GlobalRegistration.h"
#include "Engine/Registration/RegistrationTypes.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
using namespace Engine::Registration;

TEST(Registration, VoxelDownsampleReducesAndKeepsExtent) {
    PointCloud in;
    for (int i = 0; i < 40; ++i)
        for (int j = 0; j < 40; ++j) {
            in.points.emplace_back(i * 0.25f, j * 0.25f, 0.0f); // dense 10x10mm plane, 0.25mm spacing
            in.normals.emplace_back(0, 0, 1);
        }
    PointCloud out = DownsampleVoxel(in, 1.0f); // 1mm cells → ~10x10 = ~100 pts
    EXPECT_LT(out.points.size(), in.points.size());
    EXPECT_GT(out.points.size(), 50u);
    EXPECT_EQ(out.normals.size(), out.points.size());
    // normals preserved (all +Z)
    for (auto& n : out.normals) EXPECT_NEAR(n.z(), 1.0f, 1e-3f);
}

static Engine::Registration::PointCloud makeSphere(int n, float r) {
    Engine::Registration::PointCloud c;
    for (int i = 0; i < n; ++i) {
        float a = 2.399963f * i, z = 1.0f - 2.0f * (i + 0.5f) / n;
        float rr = std::sqrt(std::max(0.0f, 1 - z*z));
        Eigen::Vector3f d(rr*std::cos(a), rr*std::sin(a), z);
        c.points.push_back(r * d); c.normals.push_back(d); // outward normals
    }
    return c;
}

TEST(Registration, FpfhIsApproximatelyRotationInvariant) {
    auto s = makeSphere(600, 20.0f);
    Eigen::Matrix3f R = Eigen::AngleAxisf(0.7f, Eigen::Vector3f(0.3f,0.8f,0.5f).normalized()).toRotationMatrix();
    Engine::Registration::PointCloud sr = s;
    for (auto& p : sr.points) p = R * p;
    for (auto& nrm : sr.normals) nrm = R * nrm;
    auto f0 = Engine::Registration::ComputeFpfh(s,  60.0f, 100.0f);
    auto f1 = Engine::Registration::ComputeFpfh(sr, 60.0f, 100.0f);
    // point i maps to point i under R (same ordering), so descriptors should be close
    double maxdiff = 0;
    for (size_t i = 0; i < f0.size(); ++i) maxdiff = std::max<double>(maxdiff, (f0[i]-f1[i]).norm());
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
    Engine::Registration::PointCloud sr = s;
    for (auto& p : sr.points) p = R * p;
    for (auto& n : sr.normals) n = R * n;
    auto fs = Engine::Registration::ComputeFpfh(s,  60.0f, 100.0f);
    auto ft = Engine::Registration::ComputeFpfh(sr, 60.0f, 100.0f);
    auto corr = Engine::Registration::MatchFeatures(fs, ft);
    // most correspondences should be i→i (descriptor space is symmetric under R)
    int selfMatches = 0; for (auto& c : corr) if (c.srcIdx == c.tgtIdx) ++selfMatches;
    EXPECT_GT(corr.size(), 100u);
    EXPECT_GT(double(selfMatches) / corr.size(), 0.5) << selfMatches << "/" << corr.size();
}

TEST(Registration, RansacRecoversKnownTransform) {
    auto tgt = makeSphere(600, 20.0f);           // "model"
    // bumpy sphere so FPFH isn't degenerate: perturb radius by a hash
    for (size_t i = 0; i < tgt.points.size(); ++i) tgt.points[i] *= (1.0f + 0.15f*std::sin(0.7f*i));
    Eigen::Matrix3f Rgt = Eigen::AngleAxisf(0.6f, Eigen::Vector3f(0.2f,0.7f,0.6f).normalized()).toRotationMatrix();
    Eigen::Vector3f tgt_t(8.0f, -5.0f, 3.0f);
    Engine::Registration::PointCloud src = tgt;   // src = model moved by Tgt
    for (size_t i = 0; i < src.points.size(); ++i) { src.points[i] = Rgt*tgt.points[i] + tgt_t; src.normals[i] = Rgt*tgt.normals[i]; }
    Engine::Registration::RegistrationConfig cfg; cfg.voxelSize = 2.0f;
    auto res = Engine::Registration::EstimateRansac(src, tgt, cfg);  // aligns src→tgt ⇒ T ≈ [Rgt|tgt_t]^-1
    ASSERT_TRUE(res.valid);
    Eigen::Matrix4f Tgt = Eigen::Matrix4f::Identity(); Tgt.block<3,3>(0,0)=Rgt; Tgt.block<3,1>(0,3)=tgt_t;
    Eigen::Matrix4f err = res.T * Tgt;   // should be ≈ identity
    float rotErr = Eigen::AngleAxisf(Eigen::Matrix3f(err.block<3,3>(0,0))).angle();
    float trErr  = err.block<3,1>(0,3).norm();
    EXPECT_LT(rotErr, 0.1f) << "rot err rad";       // ~6°
    EXPECT_LT(trErr, 3.0f)  << "trans err mm";      // coarse RANSAC tolerance
}
