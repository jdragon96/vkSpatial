// Geometric loop closure (RegisterPointClouds) + its composition with the pose
// graph. Uses an asymmetric bumpy surface so FPFH descriptors are discriminative.
#include <gtest/gtest.h>

#include "Engine/Backend/Lie.h"
#include "Engine/Backend/LoopClosure.h"
#include "Engine/Backend/PoseGraph.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <cmath>
#include <random>
#include <vector>

using Engine::Backend::Mat6;
using Engine::Backend::PoseGraph;
using Engine::Backend::RegisterPointClouds;
using Engine::Backend::RegistrationConfig;
using Engine::Backend::SE3;
using Engine::Backend::SE3Exp;
using Engine::Backend::Vec6;
using Engine::Core::OrientedPointCloud;

namespace {

    struct Bump { float A, cx, cy, s; };
    const std::vector<Bump> kBumps = {
            {0.6f, -0.7f, 0.4f, 0.5f}, {-0.4f, 0.8f, -0.5f, 0.6f},
            {0.35f, 0.2f, 0.9f, 0.4f}, {0.5f, -1.1f, -0.8f, 0.7f}};

    float height(float x, float y) {
        float z = 0.0f;
        for (const auto &b : kBumps)
            z += b.A * std::exp(-((x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy)) / (2 * b.s * b.s));
        return z;
    }
    Eigen::Vector3f normalAt(float x, float y) {
        float dx = 0, dy = 0;
        for (const auto &b : kBumps) {
            const float e = b.A * std::exp(-((x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy)) / (2 * b.s * b.s));
            dx += e * (-(x - b.cx) / (b.s * b.s));
            dy += e * (-(y - b.cy) / (b.s * b.s));
        }
        return Eigen::Vector3f(-dx, -dy, 1.0f).normalized();
    }
    Engine::Core::OrientedPointCloud makeSurface(float x0, float x1, float y0, float y1, float step) {
        Engine::Core::OrientedPointCloud c;
        for (float x = x0; x <= x1 + 1e-4f; x += step)
            for (float y = y0; y <= y1 + 1e-4f; y += step) {
                c.points.emplace_back(x, y, height(x, y));
                c.normals.push_back(normalAt(x, y));
            }
        return c;
    }
    // Same geometry, perturbed by independent noise (a fresh "observation").
    Engine::Core::OrientedPointCloud observe(float x0, float x1, float y0, float y1, float step,
                               std::mt19937 &rng, float noiseStd) {
        Engine::Core::OrientedPointCloud c = makeSurface(x0, x1, y0, y1, step);
        std::normal_distribution<float> nz(0.0f, noiseStd);
        for (auto &p : c.points) p += Eigen::Vector3f(nz(rng), nz(rng), nz(rng));
        return c;
    }

    double rotErrDeg(const Eigen::Matrix3d &A, const Eigen::Matrix3d &B) {
        const double c = std::max(-1.0, std::min(1.0, 0.5 * ((A.transpose() * B).trace() - 1.0)));
        return std::acos(c) * 180.0 / M_PI;
    }
    double meanTransError(const std::vector<SE3> &e, const std::vector<SE3> &g) {
        double s = 0.0;
        for (size_t k = 0; k < g.size(); ++k) s += (e[k].t - g[k].t).norm();
        return s / double(g.size());
    }
    SE3 yawZ(double yaw, const Eigen::Vector3d &pos) {
        SE3 T;
        T.R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        T.t = pos;
        return T;
    }

} // namespace

// Registration recovers a known relative pose from partially-overlapping clouds.
TEST(LoopClosure, RecoversKnownTransform) {
    const float step = 0.15f;
    Vec6 xi;
    xi << 0.25, -0.15, 0.1, 0.10, 0.18, 0.22; // ~18 deg rotation + translation
    const SE3 Tgt = SE3Exp(xi);

    std::mt19937 rng(7);
    std::normal_distribution<float> nz(0.0f, 0.005f);
    Engine::Core::OrientedPointCloud src = makeSurface(-2.0f, 1.2f, -2.0f, 1.5f, step);
    Engine::Core::OrientedPointCloud raw = makeSurface(-1.2f, 2.0f, -1.5f, 2.0f, step); // shifted -> partial overlap
    Engine::Core::OrientedPointCloud tgt;
    for (size_t i = 0; i < raw.size(); ++i) {
        Eigen::Vector3d p = Tgt * raw.points[i].cast<double>();
        p += Eigen::Vector3d(nz(rng), nz(rng), nz(rng));
        tgt.points.push_back(p.cast<float>());
        tgt.normals.push_back((Tgt.R * raw.normals[i].cast<double>()).cast<float>().normalized());
    }

    RegistrationConfig cfg;
    cfg.fpfhRadius = 2.5f * step;
    cfg.inlierThreshold = 0.7f * step;
    cfg.ransacIterations = 6000;
    cfg.minInliers = 40;
    cfg.minFitness = 0.25;

    const auto r = RegisterPointClouds(src, tgt, cfg);
    EXPECT_TRUE(r.success);
    EXPECT_LT(rotErrDeg(r.T_source_to_target.R, Tgt.R), 1.5);
    EXPECT_LT((r.T_source_to_target.t - Tgt.t).norm(), 0.02);
}

// Unrelated clouds must not be reported as a confident loop closure.
TEST(LoopClosure, RejectsUnrelatedClouds) {
    const float step = 0.15f;
    Engine::Core::OrientedPointCloud src = makeSurface(-2.0f, 1.2f, -2.0f, 1.5f, step);
    Engine::Core::OrientedPointCloud flat = makeSurface(5.0f, 8.0f, 5.0f, 8.0f, step); // far, feature-poor
    for (auto &n : flat.normals) n = Eigen::Vector3f(0, 0, 1);

    RegistrationConfig cfg;
    cfg.fpfhRadius = 2.5f * step;
    cfg.inlierThreshold = 0.7f * step;
    cfg.minInliers = 40;
    cfg.minFitness = 0.5;
    EXPECT_FALSE(RegisterPointClouds(src, flat, cfg).success);
}

// End-to-end: a two-lap scan whose loop-closure edges come from geometric
// registration of revisited surface patches removes accumulated odometry drift.
TEST(LoopClosure, EndToEndDriftRemoval) {
    const int perLap = 12, laps = 2, N = perLap * laps;
    const double Rr = 5.0;
    auto gtPose = [&](int idx) {
        const double th = 2.0 * M_PI * (idx % perLap) / perLap;
        return yawZ(th + M_PI / 2.0, {Rr * std::cos(th), Rr * std::sin(th), 2.0});
    };
    std::vector<SE3> gt;
    for (int k = 0; k < N; ++k) gt.push_back(gtPose(k));

    std::mt19937 rng(2024);
    std::normal_distribution<double> tn(0.0, 0.01), rn(0.0, 0.012);
    auto noisy = [&](const SE3 &z) {
        Vec6 n;
        n << tn(rng), tn(rng), tn(rng), rn(rng), rn(rng), rn(rng);
        return SE3(z * SE3Exp(n));
    };
    std::vector<SE3> odo;
    for (int k = 0; k < N - 1; ++k) odo.push_back(noisy(gt[k].inverse() * gt[k + 1]));
    std::vector<SE3> init;
    init.push_back(gt[0]);
    for (int k = 0; k < N - 1; ++k) init.push_back(init.back() * odo[k]);
    const double before = meanTransError(init, gt);

    PoseGraph g;
    for (int k = 0; k < N; ++k) g.addVertex(init[k], /*fixed=*/k == 0);
    for (int k = 0; k < N - 1; ++k) g.addEdge(k, k + 1, odo[k]);

    // Loop closures: each lap-2 keyframe re-observes its lap-1 counterpart's patch.
    // Registration of the two observations yields the relative pose measurement.
    RegistrationConfig cfg;
    const float step = 0.2f;
    cfg.fpfhRadius = 2.5f * step;
    cfg.inlierThreshold = 0.7f * step;
    cfg.ransacIterations = 3000;
    cfg.minInliers = 25;
    cfg.minFitness = 0.4;
    std::mt19937 obsRng(99);
    int nLoops = 0;
    for (int k = perLap; k < N; ++k) {
        Engine::Core::OrientedPointCloud a = observe(-1.5f, 1.5f, -1.5f, 1.5f, step, obsRng, 0.004f); // frame k
        Engine::Core::OrientedPointCloud b = observe(-1.5f, 1.5f, -1.5f, 1.5f, step, obsRng, 0.004f); // frame k-perLap
        const auto r = RegisterPointClouds(a, b, cfg); // maps frame k -> frame (k-perLap)
        if (r.success) {
            g.addEdge(k - perLap, k, r.T_source_to_target); // Z_{(k-perLap)->k}
            ++nLoops;
        }
    }

    const auto rep = g.optimize();
    const double after = meanTransError(g.poses(), gt);

    EXPECT_GE(nLoops, perLap - 1);            // essentially every revisit detected
    EXPECT_LT(rep.finalChi2, rep.initialChi2);
    EXPECT_GT(before, 0.1);                   // there was real drift
    EXPECT_LT(after, before / 4.0);           // loop closure removed most of it
    EXPECT_LT(after, 0.06);
}
