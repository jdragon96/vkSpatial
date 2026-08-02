#include "Alignment.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>
#include <vector>

using Engine::Spatial::AdvancedEntry;
using Eigen::Vector3f;
using pipeline::AlignmentRegistry;
using pipeline::Frame;
using pipeline::ModelSnapshot;

namespace {
    // A 3-plane corner model snapshot (entries = centre + normal), constrains all 6 DoF.
    ModelSnapshot makeCornerModel() {
        ModelSnapshot m;
        const int half = 12;
        const float step = 0.02f;
        auto add = [&](const Vector3f &c, const Vector3f &n) {
            AdvancedEntry e;
            e.center = c;
            e.normal = n;
            e.weight = 1.0f;
            m.entries.push_back(e);
        };
        for (int i = 0; i <= half; ++i)
            for (int j = 0; j <= half; ++j) {
                const float a = i * step, b = j * step;
                add({a, b, 0.0f}, {0, 0, 1});
                add({a, 0.0f, b}, {0, 1, 0});
                add({0.0f, a, b}, {1, 0, 0});
            }
        return m;
    }
} // namespace

TEST(Alignment, IdentityReturnsIdentity) {
    pipeline::IdentityAlignment cmd;
    Frame f;
    const auto r = cmd.Execute(f, nullptr, Eigen::Isometry3f::Identity());
    EXPECT_TRUE(r.valid);
    EXPECT_TRUE(r.pose.matrix().isApprox(Eigen::Matrix4f::Identity()));
    EXPECT_STREQ(cmd.Name(), "identity");
}

TEST(Alignment, RegistryCreatesKnownAndRejectsUnknown) {
    const AlignmentRegistry reg = AlignmentRegistry::Default();
    EXPECT_TRUE(reg.Has("identity"));
    ASSERT_NE(reg.Create("identity"), nullptr);
    ASSERT_NE(reg.Create("icp"), nullptr);
    ASSERT_NE(reg.Create("global"), nullptr);
    EXPECT_EQ(reg.Create("bogus"), nullptr);
    EXPECT_STREQ(reg.Create("icp")->Name(), "icp");
}

// The ICP command aligns a perturbed frame to the model, recovering the transform.
TEST(Alignment, IcpCommandRecoversTransform) {
    const ModelSnapshot model = makeCornerModel();

    Eigen::Matrix4f known = Eigen::Matrix4f::Identity();
    known.block<3, 3>(0, 0) =
            Eigen::AngleAxisf(0.06f, Vector3f(0.2f, 1.0f, 0.4f).normalized()).toRotationMatrix();
    known.block<3, 1>(0, 3) = Vector3f(0.02f, -0.015f, 0.02f);

    Frame f;
    for (const AdvancedEntry &e : model.entries)
        f.pts.push_back((known * e.center.homogeneous()).head<3>());

    pipeline::PointToPlaneIcpAlignment cmd; // default IcpParams (maxCorrDist 0.1)
    const auto r = cmd.Execute(f, &model, Eigen::Isometry3f::Identity());
    ASSERT_TRUE(r.valid);

    // pose should map the perturbed frame points back onto the model (≈ known^{-1}).
    double rms = 0.0;
    for (std::size_t i = 0; i < f.pts.size(); ++i) {
        const Vector3f p = (r.pose * f.pts[i]);
        rms += (p - model.entries[i].center).squaredNorm();
    }
    rms = std::sqrt(rms / double(f.pts.size()));
    EXPECT_LT(rms, 2e-3);
}

// ICP with no model falls back to the prior pose (valid=false, bootstrap).
TEST(Alignment, IcpWithoutModelFallsBackToPrior) {
    pipeline::PointToPlaneIcpAlignment cmd;
    Frame f;
    f.pts.push_back({0, 0, 0});
    Eigen::Isometry3f prior = Eigen::Isometry3f::Identity();
    prior.translation() = Vector3f(1, 2, 3);
    const auto r = cmd.Execute(f, nullptr, prior);
    EXPECT_FALSE(r.valid);
    EXPECT_TRUE(r.pose.translation().isApprox(Vector3f(1, 2, 3)));
}
