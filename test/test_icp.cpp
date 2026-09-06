#include "Registration/Frontend/PointToPlaneIcp.h"
#include "Registration/RegistrationTypes.h"
#include "Common/PointCloud.h"
#include "Registration/RegistrationParam.h"

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <cmath>
#include <vector>

using Eigen::Vector3f;
using Registration::AlignPointToPlaneIcp;
using Common::PointCloud;
using Registration::RegistrationParam;

namespace {

    // A 3-plane box corner (constrains all 6 DoF for point-to-plane); target carries face normals.
    Common::PointCloud makeCorner() {
        Common::PointCloud c;
        const int half = 12;
        const float step = 0.02f;
        for (int i = 0; i <= half; ++i)
            for (int j = 0; j <= half; ++j) {
                const float a = i * step, b = j * step;
                c.points.emplace_back(a, b, 0.0f); // z=0 face, +Z normal
                c.normals.emplace_back(0, 0, 1);
                c.points.emplace_back(a, 0.0f, b); // y=0 face, +Y normal
                c.normals.emplace_back(0, 1, 0);
                c.points.emplace_back(0.0f, a, b); // x=0 face, +X normal
                c.normals.emplace_back(1, 0, 0);
            }
        return c;
    }
} // namespace

// ICP recovers the transform mapping a perturbed source cloud back onto the target surface.
TEST(Icp, RecoversKnownTransform) {
    const Common::PointCloud tgt = makeCorner();

    // Known small SE(3): ~4 deg about a tilted axis + a small translation.
    Eigen::Matrix4f known = Eigen::Matrix4f::Identity();
    known.block<3, 3>(0, 0) =
            Eigen::AngleAxisf(0.07f, Vector3f(0.3f, 1.0f, 0.5f).normalized()).toRotationMatrix();
    known.block<3, 1>(0, 3) = Vector3f(0.03f, -0.02f, 0.025f);

    // src = known * tgt (so the recovered T should be ~ known^{-1}).
    std::vector<Vector3f> src;
    src.reserve(tgt.points.size());
    for (const Vector3f &p: tgt.points)
        src.push_back((known * p.homogeneous()).head<3>());

    Registration::RegistrationParam params;
    params.maxCorrDist = 0.1f;
    params.maxIters = 30;
    params.minInliers = 20;

    const auto res = AlignPointToPlaneIcp(src, {}, tgt, Eigen::Matrix4f::Identity(), params);
    ASSERT_TRUE(res.valid);

    // Applying T to src should land on the target: residual near zero.
    const Eigen::Matrix4f T = res.T;
    double rms = 0.0;
    for (const Vector3f &s: src) {
        const Vector3f p = (T * s.homogeneous()).head<3>();
        const Vector3f orig = (known.inverse() * s.homogeneous()).head<3>();
        rms += (p - orig).squaredNorm();
    }
    rms = std::sqrt(rms / double(src.size()));
    EXPECT_LT(rms, 1e-3) << "ICP did not converge to the known inverse transform";

    // T should approximate known^{-1}.
    const Eigen::Matrix4f err = T * known - Eigen::Matrix4f::Identity();
    EXPECT_LT(err.norm(), 5e-2);
}

// Empty / normal-less target is rejected (valid=false, prior returned unchanged).
TEST(Icp, RejectsBadTarget) {
    std::vector<Vector3f> src{{0, 0, 0}, {1, 0, 0}};
    Common::PointCloud tgt; // no points
    const auto res = AlignPointToPlaneIcp(src, {}, tgt, Eigen::Matrix4f::Identity());
    EXPECT_FALSE(res.valid);
    EXPECT_TRUE(res.T.isApprox(Eigen::Matrix4f::Identity()));
}
