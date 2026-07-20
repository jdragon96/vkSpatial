// Pose-graph backend validation: build a loop trajectory, corrupt the odometry
// so the chained estimate drifts, add a loop-closure constraint, and check that
// optimisation pulls the estimate back onto the ground truth. This is the
// "synthetic drift -> backend recovers" test the backend exists to pass.
#include <gtest/gtest.h>

#include "Engine/Backend/Lie.h"
#include "Engine/Backend/PoseGraph.h"

#include <random>
#include <vector>

using Engine::Backend::Mat6;
using Engine::Backend::PoseGraph;
using Engine::Backend::SE3;
using Engine::Backend::SE3Exp;
using Engine::Backend::SE3Log;
using Engine::Backend::Vec6;

namespace {

    // Rotation about Z by yaw (radians).
    SE3 yawPose(double yaw, const Eigen::Vector3d &pos) {
        SE3 T;
        T.R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T.t = pos;
        return T;
    }

    // Mean translation error between an estimate and ground truth.
    double meanTransError(const std::vector<SE3> &est, const std::vector<SE3> &gt) {
        double s = 0.0;
        for (size_t k = 0; k < gt.size(); ++k) s += (est[k].t - gt[k].t).norm();
        return s / static_cast<double>(gt.size());
    }

} // namespace

// Lie sanity: Exp then Log is the identity round-trip.
TEST(PoseGraphLie, ExpLogRoundTrip) {
    Vec6 xi;
    xi << 0.3, -0.2, 0.5, 0.1, -0.4, 0.25; // arbitrary translation + rotation
    const Vec6 back = SE3Log(SE3Exp(xi));
    EXPECT_LT((back - xi).norm(), 1e-9);
}

// A perfect measurement against perfect poses yields a zero residual, so a graph
// initialised at ground truth has ~zero chi2.
TEST(PoseGraph, ZeroResidualAtGroundTruth) {
    PoseGraph g;
    SE3 a = yawPose(0.0, {0, 0, 0});
    SE3 b = yawPose(0.3, {1, 0.5, 0});
    g.addVertex(a, /*fixed=*/true);
    g.addVertex(b);
    g.addEdge(0, 1, a.inverse() * b); // exact relative measurement
    EXPECT_LT(g.chi2(), 1e-12);
}

// The headline test: loop closure removes accumulated odometry drift.
// Models a realistic scan that passes over the same arch twice, so every
// second-lap keyframe is recognised against its first-lap counterpart (same
// ground-truth pose) and contributes a loop-closure edge.
TEST(PoseGraph, LoopClosureRemovesDrift) {
    const int perLap = 12;
    const int laps = 2;
    const int N = perLap * laps; // lap 2 revisits lap 1's poses
    const double R = 5.0;

    // Ground-truth poses on a circle, each yawed tangent to the path; lap 2
    // keyframe k coincides with lap 1 keyframe (k - perLap).
    auto circlePose = [&](int idx) {
        const double th = 2.0 * M_PI * (idx % perLap) / perLap;
        return yawPose(th + M_PI / 2.0, {R * std::cos(th), R * std::sin(th), 0.0});
    };
    std::vector<SE3> gt;
    for (int k = 0; k < N; ++k) gt.push_back(circlePose(k));

    // Corrupt each measurement with small zero-mean noise; chaining it makes the
    // open-loop estimate drift as a random walk.
    std::mt19937 rng(12345);
    std::normal_distribution<double> tn(0.0, 0.01);  // 1 cm translation noise
    std::normal_distribution<double> rn(0.0, 0.012); // ~0.7 deg rotation noise
    auto noisy = [&](const SE3 &z) {
        Vec6 n;
        n << tn(rng), tn(rng), tn(rng), rn(rng), rn(rng), rn(rng);
        return SE3(z * SE3Exp(n));
    };

    // Consecutive odometry.
    std::vector<SE3> odo;
    for (int k = 0; k < N - 1; ++k) odo.push_back(noisy(gt[k].inverse() * gt[k + 1]));

    // Initial estimate: chain the noisy odometry from the (fixed) first pose.
    std::vector<SE3> init;
    init.push_back(gt[0]);
    for (int k = 0; k < N - 1; ++k) init.push_back(init.back() * odo[k]);

    const double driftBefore = meanTransError(init, gt);

    PoseGraph g;
    for (int k = 0; k < N; ++k) g.addVertex(init[k], /*fixed=*/k == 0);
    for (int k = 0; k < N - 1; ++k) g.addEdge(k, k + 1, odo[k]);
    for (int k = perLap; k < N; ++k) // loop closures from place recognition
        g.addEdge(k, k - perLap, noisy(gt[k].inverse() * gt[k - perLap]));

    const auto rep = g.optimize();

    const double driftAfter = meanTransError(g.poses(), gt);

    // chi2 must drop, and the trajectory must be pulled back near ground truth.
    EXPECT_LT(rep.finalChi2, rep.initialChi2);
    EXPECT_GT(driftBefore, 0.1);               // there really was drift to remove
    EXPECT_LT(driftAfter, driftBefore / 4.0);  // loop closure removes most of it
    EXPECT_LT(driftAfter, 0.06);               // and lands within 6 cm of truth
}
